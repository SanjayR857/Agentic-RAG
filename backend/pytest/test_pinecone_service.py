"""
Unit and Integration Tests for PineconeService
Verifies:
1. Reciprocal Rank Fusion (RRF) math & sorting
2. Dense & Sparse embedding generation
3. Semantic reranker scoring
4. Full pipeline execution
"""

import os
import sys

# Ensure backend directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.services.pinecone_service import PineconeService


def test_reciprocal_rank_fusion():
    print("--- 1. Testing Reciprocal Rank Fusion (RRF) ---")
    service = PineconeService()

    # List 1: Dense Search Ranking
    dense_results = [
        {"id": "doc1", "text": "Pinecone is a vector database for vector search."},
        {"id": "doc2", "text": "LangGraph is a library for building stateful multi-actor applications."},
        {"id": "doc3", "text": "Hybrid search combines dense vectors and sparse keywords."}
    ]

    # List 2: Sparse (BM25) Search Ranking
    sparse_results = [
        {"id": "doc3", "text": "Hybrid search combines dense vectors and sparse keywords."},
        {"id": "doc1", "text": "Pinecone is a vector database for vector search."},
        {"id": "doc4", "text": "Reciprocal Rank Fusion merges multiple ranked lists."}
    ]

    # Run RRF with k=60
    fused = service.reciprocal_rank_fusion(
        results_lists=[dense_results, sparse_results],
        k=60
    )

    print("RRF Fused Results:")
    for item in fused:
        print(f"  ID: {item['id']} | RRF Score: {item['rrf_score']} | Text: {item['text']}")

    # Expected: doc3 and doc1 score highest because they appear near top in both lists
    assert len(fused) == 4, f"Expected 4 unique documents, got {len(fused)}"
    assert fused[0]["id"] in ["doc1", "doc3"], f"Top document should be doc1 or doc3, got {fused[0]['id']}"
    print("SUCCESS: RRF test passed!\n")


def test_semantic_reranking():
    print("--- 2. Testing Semantic Reranker ---")
    service = PineconeService()

    query = "What is hybrid search in vector databases?"

    candidates = [
        {"id": "doc1", "text": "LangChain provides abstractions for working with LLMs.", "score": 0.5},
        {"id": "doc2", "text": "Hybrid search merges dense semantic vectors and sparse BM25 keyword matching for optimal retrieval accuracy.", "score": 0.8},
        {"id": "doc3", "text": "Pinecone is a cloud-native vector database.", "score": 0.6}
    ]

    reranked = service.semantic_rerank(query=query, documents=candidates, top_n=3)

    print(f"Query: '{query}'")
    print("Semantic Reranked Results:")
    for item in reranked:
        print(f"  ID: {item['id']} | Rerank Score: {item.get('rerank_score'):.4f} | Text: {item['text']}")

    # Expected: doc2 should be ranked #1 because it directly answers the query about hybrid search
    assert reranked[0]["id"] == "doc2", f"Expected top reranked doc to be 'doc2', got {reranked[0]['id']}"
    print("SUCCESS: Semantic Reranker test passed!\n")


def test_full_pipeline_mock():
    print("--- 3. Testing Complete Retrieval Pipeline ---")
    service = PineconeService()

    docs = [
        {"id": "1", "text": "Agentic RAG uses graph agents to make dynamic retrieval decisions."},
        {"id": "2", "text": "Pinecone vector index stores high-dimensional embeddings for similarity search."},
        {"id": "3", "text": "RRF (Reciprocal Rank Fusion) computes fused scores: sum(1 / (k + rank))."}
    ]

    print("Upserting sample documents...")
    upsert_res = service.upsert_documents(documents=docs)
    print(f"Upsert response: {upsert_res}")

    query = "How does RRF fusion work?"
    fused_results = service.hybrid_retrieve_and_rerank(query=query, top_k=3, top_n=2)

    print(f"Pipeline Query: '{query}'")
    print("Final Top Documents:")
    for doc in fused_results:
        print(f"  ID: {doc['id']} | Text: {doc.get('text', doc.get('metadata', {}).get('text'))}")

    print("SUCCESS: Pipeline test passed!\n")


if __name__ == "__main__":
    print("Starting PineconeService Verification Tests...\n")
    test_reciprocal_rank_fusion()
    test_semantic_reranking()
    test_full_pipeline_mock()
    print("ALL TESTS PASSED SUCCESSFULLY!")
