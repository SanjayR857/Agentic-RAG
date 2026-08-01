import streamlit as st
import time
from backend.app.agent.workflow import app

# Set page config
st.set_page_config(
    page_title="Agentic-RAG Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern premium feel
st.markdown("""
<style>
    .main {
        background-color: #0f1117;
        color: #e2e8f0;
    }
    .stTextInput>div>div>input {
        background-color: #1e293b;
        color: #f8fafc;
        border: 1px solid #334155;
    }
    .reportview-container .markdownTable {
        color: #f8fafc;
    }
    h1, h2, h3 {
        color: #38bdf8 !important;
        font-family: 'Inter', sans-serif;
    }
    .stExpander {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Agentic-RAG Workflow Explorer")
st.markdown("Interact with the **Hybrid RAG + CRAG (Corrective RAG) + Web Search Fallback** agent. Watch the workflow execute in real-time below!")

# Sidebar information
with st.sidebar:
    st.header("Pipeline Configurations")
    st.markdown("""
    - **Vectorstore:** Pinecone (Hybrid Sparse/Dense)
    - **Dense Model:** `embeddinggemma:300m-qat-q4_0`
    - **Sparse Model:** BM25 (pinecone-text)
    - **Reranker:** CrossEncoder (`ms-marco-MiniLM-L-6-v2`)
    - **Orchestration:** LangGraph (StateGraph)
    - **LLM Engine:** Local Ollama (`llama3`)
    - **Fallback:** Live DuckDuckGo Web Search
    """)
    st.info("Ask about recruitment policies to trigger RAG, or general knowledge to trigger Web Search routing.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "steps" in message:
            with st.expander("Explore Agent Execution Path", expanded=False):
                for step in message["steps"]:
                    st.markdown(step)

# Input box for query
if prompt := st.chat_input("Enter your question here..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process response
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        log_placeholder = st.container()
        answer_placeholder = st.empty()
        
        steps = []
        final_answer = ""
        
        with st.status("Agent thinking and executing nodes...", expanded=True) as status_box:
            try:
                # Execute graph stream
                inputs = {
                    "user_query": prompt,
                    "documents": [],
                    "web_search_needed": False,
                    "web_search_queries": [],
                    "generation": "",
                    "web_search_count": 0
                }
                
                for event in app.stream(inputs, config={"configurable": {"thread_id": "st-session"}}):
                    for node_name, state in event.items():
                        st.write(f"⚙️ **Executed Node:** `{node_name}`")
                        steps.append(f"⚙️ **Executed Node:** `{node_name}`")
                        
                        if node_name == "retrieve_rag":
                            docs = state.get("documents", [])
                            st.info(f"📚 Retrieved {len(docs)} documents from Pinecone index.")
                            steps.append(f"📚 Retrieved {len(docs)} documents.")
                            with st.expander("Retrieved Document Context Snippets", expanded=False):
                                for idx, doc in enumerate(docs):
                                    text = doc.get("text", "") or doc.get("metadata", {}).get("text", "")
                                    score = doc.get("rerank_score", 0.0)
                                    st.markdown(f"**Doc {idx+1} (Rerank Score: {score:.4f}):**\n{text[:250]}...")
                                    
                        elif node_name == "grade_documents":
                            web_search_needed = state.get("web_search_needed", False)
                            relevant_count = len(state.get("documents", []))
                            if web_search_needed:
                                st.warning("⚠️ All retrieved documents graded as IRRELEVANT. Triggering web search.")
                                steps.append("⚠️ CRAG Graded all documents as irrelevant.")
                            else:
                                st.success(f"✅ Found {relevant_count} relevant documents. Ready to generate.")
                                steps.append(f"✅ CRAG Graded {relevant_count} documents as relevant.")
                                
                        elif node_name == "web_search":
                            search_count = state.get("web_search_count", 0)
                            docs = state.get("documents", [])
                            st.info(f"🌐 Querying DuckDuckGo (Retries: {search_count}/3).")
                            steps.append(f"🌐 Queried DuckDuckGo web search. Total documents in context: {len(docs)}.")
                            
                        elif node_name == "generate":
                            final_answer = state.get("generation", "")
                            steps.append("🤖 Generated candidate answer using Ollama.")
                            
                            # Let's show the evaluation logs since generate node output routes next
                            # (Wait a split second to make the UI transition smooth)
                            time.sleep(0.5)
                            
                status_box.update(label="Workflow completed successfully!", state="complete", expanded=False)
                
            except Exception as e:
                status_box.update(label="Execution failed!", state="error")
                st.error(f"Error during agent execution: {e}")
                final_answer = "Sorry, I encountered an error while processing your request."
        
        # Display final answer
        if final_answer:
            answer_placeholder.markdown(final_answer)
            # Add assistant message to session history
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "steps": steps
            })
