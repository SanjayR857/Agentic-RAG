from langgraph.graph import StateGraph, END
from backend.app.agent.state import AgentState
from backend.app.agent.node import (
    supervisor_agent,
    retrieve_rag,
    grade_documents,
    route_crag,
    web_search,
    generate,
    route_generation
)

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("retrieve_rag", retrieve_rag)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

# Set the conditional entry point using our supervisor_agent function
workflow.set_conditional_entry_point(
    supervisor_agent,
    {
        "rag": "retrieve_rag",
        "web_search": "web_search"
    }
)

# Connect nodes
workflow.add_edge("retrieve_rag", "grade_documents")

# Set conditional edge after grading documents
workflow.add_conditional_edges(
    "grade_documents",
    route_crag,
    {
        "web_search": "web_search",
        "generate": "generate"
    }
)

# web_search proceeds to generate
workflow.add_edge("web_search", "generate")

# Set conditional edge after answer generation
workflow.add_conditional_edges(
    "generate",
    route_generation,
    {
        "web_search": "web_search",
        "useful": END
    }
)

# Compile the workflow
app = workflow.compile()
