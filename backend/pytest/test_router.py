import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / "backend" / ".env")

from backend.app.agent.workflow import app

def test_routing(query: str):
    print(f"\n--- Testing Query: '{query}' ---")
    inputs = {
        "user_query": query,
        "documents": [],
        "web_search_needed": False,
        "web_search_queries": [],
        "generation": "",
        "web_search_count": 0
    }
    
    # Run the compiled StateGraph workflow
    try:
        events = app.stream(inputs)
        for event in events:
            for node_name, state in event.items():
                print(f"Graph executed node: '{node_name}' (Web Search Retries: {state.get('web_search_count', 0)})")
                if "documents" in state and state["documents"] and node_name == "retrieve_rag":
                    print(f"Retrieved {len(state['documents'])} documents:")
                    for idx, doc in enumerate(state["documents"]):
                        text = doc.get("text", "") or doc.get("metadata", {}).get("text", "")
                        print(f"  Doc {idx+1}: {text[:100]}... (Rerank Score: {doc.get('rerank_score', 0.0)})")
                if "generation" in state and state["generation"] and node_name == "generate":
                    print(f"  Generation: {state['generation']}")
    except Exception as e:
        print(f"Workflow stream error: {e}")

if __name__ == "__main__":
    test_queries = [
        "what is python?"
    ]
    
    for q in test_queries:
        test_routing(q)
