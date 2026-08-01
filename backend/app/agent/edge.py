import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from backend.app.agent.state import AgentState

class RouteDecision(BaseModel):
    datasource: str = Field(
        description="The data source to route the query to. Either 'rag' or 'web_search'."
    )

class GradeHallucination(BaseModel):
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
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
    
    model_name = os.getenv("OLLAMA_ROUTER_MODEL", "llama3")
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
        
    model_name = os.getenv("OLLAMA_GRADER_MODEL", "llama3")
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
        
    if web_search_count < 3:
        print(f"Check failed. Retry count: {web_search_count}/3. Routing to web_search.")
        return "web_search"
    else:
        print("Check failed but maximum web search retries reached. Ending workflow.")
        return "useful"
