"""
Pinecone Service Module

Provides complete integration with Pinecone for vector indexing, hybrid retrieval
(dense vector search + BM25 sparse search), Reciprocal Rank Fusion (RRF),
and Semantic Reranking.
"""

import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import SentenceTransformer, CrossEncoder
from backend.app.core.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PineconeService")


class PineconeService:
    """
    Pinecone Service encapsulating:
    1. Index Creation & Management
    2. Dense Vector & BM25 Sparse Vector Upserts
    3. Dense, Sparse, and Hybrid Retrieval
    4. Reciprocal Rank Fusion (RRF)
    5. Semantic Reranking (Cross-Encoder / Pinecone Rerank)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        reranker_model_name: str = settings.CROSS_ENCODER_MODEL,
        dimension: int = 768,
        metric: str = "dotproduct"
    ):
        """
        Initialize PineconeService.
        
        Args:
            api_key: Pinecone API key. Defaults to PINECONE_API_KEY env var.
            index_name: Name of Pinecone index. Defaults to PINECONE_INDEX_NAME env var or 'agentic-rag-index'.
            embedding_model_name: Embedding model (e.g., 'embeddinggemma:300m-qat-q4_0' or 'all-MiniLM-L6-v2').
            reranker_model_name: HuggingFace CrossEncoder model for semantic reranking.
            dimension: Vector dimension for embedding model (768 for embeddinggemma:300m-qat-q4_0).
            metric: Distance metric for index ('dotproduct' recommended for hybrid search).
        """
        self.api_key = api_key or settings.PINECONE_API_KEY
        self.index_name = index_name or settings.PINECONE_INDEX_NAME
        self.embedding_model_name = (embedding_model_name or settings.OLLAMA_EMBEDDING_MODEL).strip()
        self.dimension = dimension
        self.metric = metric

        # Clients and Models
        self.pc: Optional[Any] = None
        self.index: Optional[Any] = None
        self.ollama_embeddings: Optional[Any] = None
        self.embedding_model: Optional[Any] = None
        self.reranker_model: Optional[Any] = None
        self.bm25_encoder: Optional[Any] = None

        # Initialize Models & Services
        self._init_models(self.embedding_model_name, reranker_model_name)
        self._init_pinecone()

    def _init_models(self, embedding_model_name: str, reranker_model_name: str):
        """Initialize embedding, sparse, and reranker models."""
        try:
            logger.info(f"Loading OllamaEmbeddings model: '{embedding_model_name}'")
            self.ollama_embeddings = OllamaEmbeddings(model=embedding_model_name)
        except Exception as e:
            logger.warning(f"Could not load OllamaEmbeddings: {e}")

        if not self.ollama_embeddings:
            logger.info(f"Loading SentenceTransformer dense embedding model: {embedding_model_name}")
            try:
                self.embedding_model = SentenceTransformer(embedding_model_name)
            except Exception:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.dimension = 384

        logger.info(f"Loading semantic reranker model: {reranker_model_name}")
        self.reranker_model = CrossEncoder(reranker_model_name)

        logger.info("Initializing default BM25Encoder for sparse retrieval.")
        self.bm25_encoder = BM25Encoder.default()

    def _init_pinecone(self):
        """Initialize Pinecone client and index if API key is provided."""
        if not self.api_key:
            logger.warning("No PINECONE_API_KEY found. Operating in offline/mock mode.")
            return

        try:
            self.pc = Pinecone(api_key=self.api_key)
            self.index = self.get_or_create_index(self.index_name, dimension=self.dimension, metric=self.metric)
            logger.info(f"Successfully connected to Pinecone index: '{self.index_name}'")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client/index: {e}")
            logger.info(f"Successfully connected to Pinecone index: '{self.index_name}'")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client/index: {e}")

    def get_or_create_index(
        self,
        index_name: str,
        dimension: int = 384,
        metric: str = "dotproduct",
        cloud: str = "aws",
        region: str = "us-east-1"
    ) -> Any:
        """
        Get an existing Pinecone index or create a new Serverless index if it doesn't exist.
        
        Args:
            index_name: Name of index.
            dimension: Dimension of vectors.
            metric: Distance metric ('cosine', 'dotproduct', 'euclidean').
            cloud: Serverless cloud provider ('aws', 'gcp', 'azure').
            region: Serverless cloud region ('us-east-1', etc.).
            
        Returns:
            Pinecone Index instance.
        """
        if not self.pc:
            raise RuntimeError("Pinecone client is not initialized. Provide a valid API key.")

        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if index_name not in existing_indexes:
            logger.info(f"Creating new Pinecone Serverless Index: '{index_name}' (dim={dimension}, metric={metric})...")
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
            logger.info(f"Index '{index_name}' created successfully.")
        else:
            logger.info(f"Index '{index_name}' already exists.")

        return self.pc.Index(index_name)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embeddings for a list of text strings."""
        if self.ollama_embeddings:
            try:
                return self.ollama_embeddings.embed_documents(texts)
            except Exception as e:
                logger.error(f"Error generating embeddings via OllamaEmbeddings: {e}")

        if self.embedding_model:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()
        else:
            # Fallback mock embedding for testing
            return [[0.1] * self.dimension for _ in texts]

    def generate_sparse_vectors(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Generate BM25 sparse vectors for a list of text strings."""
        if self.bm25_encoder:
            return self.bm25_encoder.encode_documents(texts)
        else:
            # Fallback empty/mock sparse vectors
            return [{"indices": [1, 2, 3], "values": [0.5, 0.3, 0.2]} for _ in texts]

    def fit_bm25(self, corpus: List[str]):
        """Fit BM25 encoder on custom text corpus."""
        if self.bm25_encoder:
            logger.info(f"Fitting BM25Encoder on corpus of {len(corpus)} documents...")
            self.bm25_encoder.fit(corpus)

    def upsert_documents(
        self,
        documents: List[Dict[str, Any]],
        namespace: str = ""
    ) -> Dict[str, Any]:
        """
        Upsert documents with dense and sparse vectors into Pinecone.
        
        Args:
            documents: List of dicts with keys: 'id', 'text', and optional 'metadata'.
            namespace: Namespace inside Pinecone index.
            
        Returns:
            Pinecone upsert response summary dict.
        """
        if not documents:
            return {"upserted_count": 0}

        texts = [doc["text"] for doc in documents]
        ids = [doc["id"] for doc in documents]

        # Generate vectors
        dense_vecs = self.generate_embeddings(texts)
        sparse_vecs = self.generate_sparse_vectors(texts)

        vectors = []
        for i in range(len(documents)):
            metadata = documents[i].get("metadata", {})
            metadata["text"] = texts[i]

            vector_item = {
                "id": ids[i],
                "values": dense_vecs[i],
                "sparse_values": sparse_vecs[i],
                "metadata": metadata
            }
            vectors.append(vector_item)

        if self.index:
            response = self.index.upsert(vectors=vectors, namespace=namespace)
            logger.info(f"Upserted {len(vectors)} documents into Pinecone namespace '{namespace}'.")
            return response
        else:
            logger.info(f"[Offline Mode] Prepared {len(vectors)} vectors for upsert.")
            return {"upserted_count": len(vectors), "mode": "offline"}

    def dense_search(
        self,
        query: str,
        top_k: int = 10,
        namespace: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Perform Dense Vector Similarity Search.
        """
        query_dense = self.generate_embeddings([query])[0]

        if self.index:
            res = self.index.query(
                vector=query_dense,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace
            )
            return self._format_pinecone_hits(res.get("matches", []))
        else:
            logger.info("[Offline Mode] Returning dummy dense search results.")
            return []

    def sparse_search(
        self,
        query: str,
        top_k: int = 10,
        namespace: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Perform Sparse Vector (BM25 Keyword) Search.
        """
        if self.bm25_encoder:
            query_sparse = self.bm25_encoder.encode_queries(query)
        else:
            query_sparse = {"indices": [1, 2], "values": [0.5, 0.5]}

        if self.index:
            # Pinecone dense index queries with sparse_vector require passing a vector argument
            dummy_vector = [0.0] * self.dimension
            res = self.index.query(
                vector=dummy_vector,
                sparse_vector=query_sparse,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace
            )
            return self._format_pinecone_hits(res.get("matches", []))
        else:
            logger.info("[Offline Mode] Returning dummy sparse search results.")
            return []

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        namespace: str = ""
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform Hybrid Retrieval by retrieving top_k dense results and top_k sparse results.
        
        Args:
            query: User search query.
            top_k: Top K items to retrieve from each retriever.
            alpha: Weight for dense vs sparse (for scaling if needed).
            namespace: Index namespace.
            
        Returns:
            Dict containing 'dense_hits' and 'sparse_hits'.
        """
        dense_hits = self.dense_search(query=query, top_k=top_k, namespace=namespace)
        sparse_hits = self.sparse_search(query=query, top_k=top_k, namespace=namespace)

        return {
            "dense_hits": dense_hits,
            "sparse_hits": sparse_hits
        }

    def reciprocal_rank_fusion(
        self,
        results_lists: List[List[Dict[str, Any]]],
        k: int = 60,
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) algorithm.
        
        Combines multiple ranked result lists into a single ranked list using the formula:
        RRF_Score(doc) = sum(1 / (k + rank_m(doc))) for all rankers m.
        
        Args:
            results_lists: List of ranked document lists (each item must have 'id').
            k: Constant smoothing parameter (default: 60).
            top_n: Optional cutoff number of top fused documents to return.
            
        Returns:
            Fused list of document dicts ordered descending by 'rrf_score'.
        """
        rrf_scores: Dict[str, float] = {}
        doc_store: Dict[str, Dict[str, Any]] = {}

        for rank_list in results_lists:
            for rank, doc in enumerate(rank_list, start=1):
                doc_id = doc["id"]
                if doc_id not in doc_store:
                    doc_store[doc_id] = doc.copy()

                score_addition = 1.0 / (k + rank)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score_addition

        # Sort documents by RRF score descending
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda id_: rrf_scores[id_], reverse=True)

        fused_results = []
        for doc_id in sorted_doc_ids:
            doc = doc_store[doc_id].copy()
            doc["rrf_score"] = round(rrf_scores[doc_id], 6)
            fused_results.append(doc)

        if top_n is not None:
            fused_results = fused_results[:top_n]

        return fused_results

    def semantic_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform Semantic Re-ranking on candidate documents using a Cross-Encoder.
        
        Args:
            query: User query string.
            documents: Candidate documents (each containing 'text' or 'metadata': {'text': ...}).
            top_n: Number of top reranked documents to return.
            
        Returns:
            Top N documents sorted descending by 'rerank_score'.
        """
        if not documents:
            return []

        # Extract text from documents
        doc_texts = []
        for doc in documents:
            text = doc.get("text") or doc.get("metadata", {}).get("text", "")
            doc_texts.append(text)

        if self.reranker_model:
            # Pair query with each document text
            pairs = [[query, text] for text in doc_texts]
            scores = self.reranker_model.predict(pairs)

            # Attach rerank score to documents
            reranked_docs = []
            for i, doc in enumerate(documents):
                doc_copy = doc.copy()
                doc_copy["rerank_score"] = float(scores[i])
                reranked_docs.append(doc_copy)

            # Sort descending by rerank_score
            reranked_docs.sort(key=lambda d: d["rerank_score"], reverse=True)
            return reranked_docs[:top_n]
        elif self.pc and hasattr(self.pc, "inference"):
            # Pinecone Inference Rerank fallback if pc.inference is available
            try:
                logger.info("Using Pinecone Inference Rerank API...")
                res = self.pc.inference.rerank(
                    model="bge-reranker-large",
                    query=query,
                    documents=doc_texts,
                    top_n=top_n
                )
                reranked_docs = []
                for item in res.data:
                    idx = item.index
                    doc_copy = documents[idx].copy()
                    doc_copy["rerank_score"] = float(item.score)
                    reranked_docs.append(doc_copy)
                return reranked_docs
            except Exception as e:
                logger.warning(f"Pinecone inference rerank error: {e}. Falling back to default ordering.")

        # Fallback if no reranker model available: keep original order
        for doc in documents:
            doc.setdefault("rerank_score", doc.get("rrf_score", doc.get("score", 0.0)))
        return documents[:top_n]

    def hybrid_retrieve_and_rerank(
        self,
        query: str,
        top_k: int = 10,
        top_n: int = 5,
        namespace: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Complete Retrieval Pipeline:
        1. Hybrid Search (Dense + Sparse retrieval)
        2. Reciprocal Rank Fusion (RRF)
        3. Semantic Reranking of fused top candidates.
        """
        # Step 1: Hybrid Retrieval
        hybrid_res = self.hybrid_search(query=query, top_k=top_k, namespace=namespace)

        # Step 2: Reciprocal Rank Fusion
        fused_candidates = self.reciprocal_rank_fusion(
            results_lists=[hybrid_res["dense_hits"], hybrid_res["sparse_hits"]],
            k=60,
            top_n=top_k * 2
        )

        # Step 3: Semantic Reranking
        final_reranked = self.semantic_rerank(query=query, documents=fused_candidates, top_n=top_n)

        return final_reranked

    def _format_pinecone_hits(self, matches: List[Any]) -> List[Dict[str, Any]]:
        """Format Pinecone SDK match objects into dictionaries."""
        formatted = []
        for match in matches:
            hit = {
                "id": getattr(match, "id", match.get("id")),
                "score": float(getattr(match, "score", match.get("score", 0.0))),
                "metadata": getattr(match, "metadata", match.get("metadata", {})),
            }
            if "text" in hit["metadata"]:
                hit["text"] = hit["metadata"]["text"]
            formatted.append(hit)
        return formatted
