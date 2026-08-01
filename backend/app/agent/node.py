import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from backend.app.agent.state import AgentState
from backend.app.services.pinecone_service import PineconeService
from backend.app.core.config import settings

class RouteDecision(BaseModel):
    datasource: str = Field(
        description="The data source to route the query to. Either 'rag' or 'web_search'."
    )

def supervisor_agent(state: AgentState) -> str:
    """
    Routes the query to either RAG or Web Search.
    
    Args:
        state (AgentState): The current state containing user_query.
        
    Returns:
        str: Next node to route to ("rag" or "web_search")
    """
    user_query = state.get("user_query")
    print(f"Routing query: '{user_query}'")
    
    model_name = settings.OLLAMA_ROUTER_MODEL
    llm = ChatOllama(model=model_name, format="json", temperature=0)
    
    parser = JsonOutputParser(pydantic_object=RouteDecision)
    
    prompt = PromptTemplate(
        template=(
            "You are an expert at routing user questions to either a RAG pipeline (vectorstore) or Web Search.\n"
            "The RAG pipeline contains internal documents related to company recruitment policies, employee guidelines, internal procedures, and onboarding.\n"
            "Use RAG for any question regarding internal policy, company recruitment, benefits, guidelines, or internal systems.\n"
            "Use Web Search for general knowledge, external news, topics unrelated to the company, or programming help.\n\n"
            "{format_instructions}\n\n"
            "Question: {question}"
        ),
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    try:
        decision = chain.invoke({"question": user_query})
        datasource = decision.get("datasource", "web_search").lower()
        if datasource in ["rag", "web_search"]:
            print(f"Routing decision: {datasource}")
            return datasource
    except Exception as e:
        print(f"Router error: {e}. Defaulting to web_search.")
        
    return "web_search"



def retrieve_rag(state: AgentState) -> AgentState:
    """
    Retrieves documents from Pinecone using hybrid search + RRF + semantic reranking.
    
    Args:
        state (AgentState): The current state.
        
    Returns:
        AgentState: The updated state with retrieved documents.
    """
    user_query = state.get("user_query")
    print(f"Retrieving documents for: '{user_query}'")
    
    try:
        pinecone_service = PineconeService()
        docs = pinecone_service.hybrid_retrieve_and_rerank(
            user_query, 
            top_k=settings.RAG_RETRIEVE_TOP_K, 
            top_n=settings.RAG_RERANK_TOP_N
        )
        print(f"Retrieved {len(docs)} documents.")
        return {**state, "documents": docs, "web_search_count": state.get("web_search_count", 0)}
    except Exception as e:
        print(f"Retrieval error: {e}")
        return {**state, "documents": [], "web_search_count": state.get("web_search_count", 0)}

class GradeDocument(BaseModel):
    binary_score: str = Field(
        description="Document is relevant to the question, 'yes' or 'no'"
    )

def grade_documents(state: AgentState) -> AgentState:
    """
    Grades retrieved documents for relevance to the user query.
    If none of the documents are relevant, triggers web search fallback.
    
    Args:
        state (AgentState): The current state.
        
    Returns:
        AgentState: The updated state with relevant documents and web search flags.
    """
    user_query = state.get("user_query")
    documents = state.get("documents", [])
    print(f"--- CRAG: GRADING DOCUMENTS ---")
    
    model_name = settings.OLLAMA_GRADER_MODEL
    llm = ChatOllama(model=model_name, format="json", temperature=0)
    parser = JsonOutputParser(pydantic_object=GradeDocument)
    
    prompt = PromptTemplate(
        template=(
            "You are a grader assessing relevance of a retrieved document to a user question.\n"
            "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.\n"
            "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.\n\n"
            "{format_instructions}\n\n"
            "Document: {document}\n"
            "Question: {question}"
        ),
        input_variables=["document", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    
    relevant_docs = []
    web_search_needed = False
    
    if documents:
        payloads = [{"document": doc.get("text", "") or doc.get("metadata", {}).get("text", ""), "question": user_query} for doc in documents]
        try:
            scores = chain.batch(payloads)
            for idx, score_res in enumerate(scores):
                score = score_res.get("binary_score", "no").lower().strip()
                if score == "yes":
                    print(f"  Doc {idx+1}: RELEVANT")
                    relevant_docs.append(documents[idx])
                else:
                    print(f"  Doc {idx+1}: NOT RELEVANT")
        except Exception as e:
            print(f"Parallel grading error: {e}. Falling back to keeping all documents.")
            relevant_docs = documents
            
    # If no relevant documents, trigger web search
    if not relevant_docs:
        print("All documents graded as irrelevant. Web search needed.")
        web_search_needed = True
        web_search_queries = [user_query]
    else:
        print(f"Found {len(relevant_docs)} relevant documents. Web search not needed.")
        web_search_needed = False
        web_search_queries = []
        
    return {
        **state,
        "documents": relevant_docs,
        "web_search_needed": web_search_needed,
        "web_search_queries": web_search_queries
    }

def route_crag(state: AgentState) -> str:
    """
    Routes the workflow to either 'web_search' or 'generate' based on CRAG analysis.
    
    Args:
        state (AgentState): The current state.
        
    Returns:
        str: Next node to route to ("web_search" or "generate")
    """
    web_search_needed = state.get("web_search_needed", False)
    if web_search_needed:
        print("CRAG Routing decision: web_search")
        return "web_search"
    else:
        print("CRAG Routing decision: generate")
        return "generate"

class GradeHallucination(BaseModel):
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

def route_generation(state: AgentState) -> str:
    """
    Evaluates the generated answer for hallucinations and query relevance.
    Routes to 'web_search' if check fails and retry limit (3) is not exceeded.
    Otherwise routes to END.
    
    Args:
        state (AgentState): The current state.
        
    Returns:
        str: Next node to route to ("web_search" or "useful")
    """
    user_query = state.get("user_query")
    generation = state.get("generation", "")
    documents = state.get("documents", [])
    web_search_count = state.get("web_search_count", 0)
    
    print("--- EVALUATING GENERATED ANSWER ---")
    
    if not generation:
        print("Generation is empty. Routing to web_search.")
        return "web_search"
        
    model_name = settings.OLLAMA_GRADER_MODEL
    llm = ChatOllama(model=model_name, format="json", temperature=0)
    
    # 1. Hallucination Check
    hallucination_parser = JsonOutputParser(pydantic_object=GradeHallucination)
    hallucination_prompt = PromptTemplate(
        template=(
            "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.\n"
            "Give a binary score 'yes' or 'no'. 'yes' means the answer is grounded in and supported by the facts.\n"
            "Facts: {documents}\n"
            "Generation: {generation}\n\n"
            "{format_instructions}"
        ),
        input_variables=["documents", "generation"],
        partial_variables={"format_instructions": hallucination_parser.get_format_instructions()},
    )
    
    hallucination_chain = hallucination_prompt | llm | hallucination_parser
    
    is_grounded = True
    try:
        doc_texts = "\n".join([doc.get("text", "") or doc.get("metadata", {}).get("text", "") for doc in documents])
        res = hallucination_chain.invoke({"documents": doc_texts, "generation": generation})
        score = res.get("binary_score", "no").lower().strip()
        if score == "yes":
            print("  Hallucination Check: PASSED (Answer is grounded)")
        else:
            print("  Hallucination Check: FAILED (Answer is hallucinated)")
            is_grounded = False
    except Exception as e:
        print(f"Error during hallucination grading: {e}")
        is_grounded = False
        
    # 2. Relevance Check
    answer_parser = JsonOutputParser(pydantic_object=GradeAnswer)
    answer_prompt = PromptTemplate(
        template=(
            "You are a grader assessing whether an LLM generation addresses / answers the user question.\n"
            "Give a binary score 'yes' or 'no'. 'yes' means the answer directly addresses and answers the question.\n"
            "Question: {question}\n"
            "Generation: {generation}\n\n"
            "{format_instructions}"
        ),
        input_variables=["question", "generation"],
        partial_variables={"format_instructions": answer_parser.get_format_instructions()},
    )
    
    answer_chain = answer_prompt | llm | answer_parser
    
    is_relevant = True
    try:
        res = answer_chain.invoke({"question": user_query, "generation": generation})
        score = res.get("binary_score", "no").lower().strip()
        if score == "yes":
            print("  Relevance Check: PASSED (Answer is relevant)")
        else:
            print("  Relevance Check: FAILED (Answer is not relevant)")
            is_relevant = False
    except Exception as e:
        print(f"Error during relevance grading: {e}")
        is_relevant = False
        
    # Determine Routing
    if is_grounded and is_relevant:
        print("Generation is useful. Ending workflow.")
        return "useful"
        
    if web_search_count < settings.MAX_WEB_SEARCH_RETRIES:
        print(f"Check failed. Retry count: {web_search_count}/{settings.MAX_WEB_SEARCH_RETRIES}. Routing to web_search.")
        return "web_search"
    else:
        print("Check failed but maximum web search retries reached. Ending workflow.")
        return "useful"

def web_search(state: AgentState) -> AgentState:
    """
    Executes web search using DuckDuckGo with optimized query reformulation.
    """
    print("--- NODE: RUNNING WEB SEARCH ---")
    user_query = state.get("user_query")
    web_search_count = state.get("web_search_count", 0) + 1
    print(f"Web search count: {web_search_count}/{settings.MAX_WEB_SEARCH_RETRIES}")
    
    # 1. Optimize search query using Ollama
    print("--- NODE: OPTIMIZING SEARCH QUERY ---")
    model_name = settings.OLLAMA_ROUTER_MODEL
    llm = ChatOllama(model=model_name, temperature=0)
    prompt = f"Optimize this user question into a concise keyword search query for Google/DuckDuckGo. Only return the search query, do not write anything else.\n\nQuestion: {user_query}\nSearch Query:"
    try:
        search_query = llm.invoke(prompt).content.strip().strip('"').strip("'")
    except Exception as e:
        print(f"Error optimizing search query: {e}")
        search_query = user_query
    print(f"Optimized search query: '{search_query}'")
    
    # 2. Query DuckDuckGo
    from duckduckgo_search import DDGS
    print(f"--- NODE: QUERYING DUCKDUCKGO: '{search_query}' ---")
    web_results = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(search_query, max_results=3)
            for r in results:
                web_results.append({
                    "id": f"web_{r['href']}",
                    "text": r['body'],
                    "metadata": {
                        "title": r['title'],
                        "source": r['href']
                    }
                })
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
        
    print(f"Retrieved {len(web_results)} web search documents.")
    
    # 3. Append to existing documents
    current_docs = state.get("documents", [])
    updated_docs = current_docs + web_results
    
    return {
        **state,
        "documents": updated_docs,
        "web_search_count": web_search_count
    }

def generate(state: AgentState) -> AgentState:
    """
    Generates answer using Ollama.
    """
    print("--- NODE: GENERATING ANSWER ---")
    user_query = state.get("user_query")
    web_search_needed = state.get("web_search_needed", False)
    documents = state.get("documents", [])
    web_search_count = state.get("web_search_count", 0)
    
    # Filter documents based on whether web search was triggered
    if web_search_needed:
        docs_to_use = [doc for doc in documents if str(doc.get("id", "")).startswith("web_")]
        print(f"Using {len(docs_to_use)} web search documents for generation.")
    else:
        docs_to_use = [doc for doc in documents if not str(doc.get("id", "")).startswith("web_")]
        print(f"Using {len(docs_to_use)} RAG documents for generation.")
        
    # Simple generation using Ollama
    model_name = settings.OLLAMA_GENERATION_MODEL
    llm = ChatOllama(model=model_name, temperature=0)
    
    doc_texts = "\n\n".join([doc.get("text", "") or doc.get("metadata", {}).get("text", "") for doc in docs_to_use])
    
    prompt = (
        "You are a knowledgeable team assistant. Use the provided reference material to answer the user question.\n"
        "To answer, you MUST think step-by-step first inside <thinking> tags, and then provide your final response outside the tags.\n\n"
        "Format your output exactly like this:\n"
        "<thinking>\n"
        "[Briefly list the facts found and determine the answer step-by-step]\n"
        "</thinking>\n"
        "[Your final friendly, detailed, and professional answer here]\n\n"
        "Requirements for the final answer:\n"
        "- Provide a comprehensive, detailed, and thorough explanation covering all aspects of the user's question.\n"
        "- Use structured formatting like bullet points, lists, and bold text to present advantages, disadvantages, or use cases clearly.\n"
        "- Ensure the explanation is complete and detailed rather than keeping it too short.\n\n"
        f"Reference Material:\n{doc_texts}\n\n"
        f"User Question: {user_query}\n\n"
        "Response:"
    )
    
    # We can inject a mock hallucination during tests to verify retry loop, e.g. if query has "hallucinate"
    if "hallucinate" in user_query.lower():
        generation = "This is a completely hallucinated answer about cats."
    else:
        try:
            raw_generation = llm.invoke(prompt).content
            if "</thinking>" in raw_generation:
                generation = raw_generation.split("</thinking>")[-1].strip()
            else:
                generation = raw_generation.strip()
        except Exception as e:
            print(f"Generation error: {e}")
            generation = "Unable to generate answer."
            
    print(f"Generated answer: {generation[:100]}...")
    return {
        **state,
        "generation": generation,
        "web_search_count": web_search_count
    }
