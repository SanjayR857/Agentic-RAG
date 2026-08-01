from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    """
    Agent State for Hybrid RAG + CRAG (Corrective RAG) Workflow:
    - user_query: Original prompt from user
    - documents: Retrieved documents (with text, metadata, RRF score, rerank score)
    - web_search_needed: Flag set by CRAG evaluation node when documents are irrelevant
    - web_search_queries: Search query string(s) generated for web fallback
    - generation: Final LLM answer synthesis
    """
    user_query: str
    documents: List[Dict[str, Any]]
    web_search_needed: bool
    web_search_queries: List[str]
    generation: str
    web_search_count: int
