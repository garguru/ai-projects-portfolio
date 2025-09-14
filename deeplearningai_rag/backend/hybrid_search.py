"""
Hybrid Search implementation combining BM25 keyword search with semantic search.
This module implements advanced retrieval techniques for improved RAG performance.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class HybridSearch:
    """
    Implements hybrid search combining BM25 keyword search with semantic search.
    Uses Reciprocal Rank Fusion (RRF) for merging results.
    """

    def __init__(
        self,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        rrf_k: int = 60,
    ):
        """
        Initialize the hybrid search system.

        Args:
            semantic_weight: Weight for semantic search scores (0-1)
            keyword_weight: Weight for keyword search scores (0-1)
            rrf_k: Constant for RRF calculation (typically 60)
        """
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k
        self.bm25_index = None
        self.documents = []
        self.scaler = MinMaxScaler()

    def build_bm25_index(self, documents: List[str]) -> None:
        """
        Build BM25 index from documents.

        Args:
            documents: List of document texts
        """
        # Tokenize documents for BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25_index = BM25Okapi(tokenized_docs)
        self.documents = documents
        logger.info(f"Built BM25 index with {len(documents)} documents")

    def keyword_search(
        self, query: str, top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Perform BM25 keyword search.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (document_index, score) tuples
        """
        if self.bm25_index is None:
            logger.warning("BM25 index not built, returning empty results")
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)

        # Get top-k indices and scores
        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = [(int(idx), float(scores[idx])) for idx in top_indices]

        return results

    def reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple[str, float]],
        keyword_results: List[Tuple[int, float]],
        doc_id_mapping: Optional[Dict[int, str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).

        Args:
            semantic_results: List of (doc_id, score) from semantic search
            keyword_results: List of (doc_index, score) from BM25
            doc_id_mapping: Mapping from doc indices to IDs

        Returns:
            Merged and reranked results
        """
        rrf_scores = {}

        # Process semantic search results
        for rank, (doc_id, score) in enumerate(semantic_results):
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (
                rrf_score * self.semantic_weight
            )

        # Process keyword search results
        for rank, (doc_idx, score) in enumerate(keyword_results):
            # Map index to doc_id if mapping provided
            doc_id = str(doc_idx)
            if doc_id_mapping:
                doc_id = doc_id_mapping.get(doc_idx, str(doc_idx))

            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (
                rrf_score * self.keyword_weight
            )

        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_results

    def weighted_fusion(
        self,
        semantic_results: List[Tuple[str, float]],
        keyword_results: List[Tuple[int, float]],
        doc_id_mapping: Optional[Dict[int, str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Merge results using weighted score fusion.

        Args:
            semantic_results: List of (doc_id, score) from semantic search
            keyword_results: List of (doc_index, score) from BM25
            doc_id_mapping: Mapping from doc indices to IDs

        Returns:
            Merged and reranked results
        """
        combined_scores = {}

        # Normalize semantic scores
        if semantic_results:
            semantic_scores = np.array([s for _, s in semantic_results])
            if semantic_scores.max() > 0:
                semantic_scores = semantic_scores / semantic_scores.max()

            for (doc_id, _), norm_score in zip(semantic_results, semantic_scores):
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + (
                    norm_score * self.semantic_weight
                )

        # Normalize keyword scores
        if keyword_results:
            keyword_scores = np.array([s for _, s in keyword_results])
            if keyword_scores.max() > 0:
                keyword_scores = keyword_scores / keyword_scores.max()

            for (doc_idx, _), norm_score in zip(keyword_results, keyword_scores):
                doc_id = str(doc_idx)
                if doc_id_mapping:
                    doc_id = doc_id_mapping.get(doc_idx, str(doc_idx))

                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + (
                    norm_score * self.keyword_weight
                )

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_results

    def hybrid_search(
        self,
        query: str,
        semantic_results: List[Tuple[str, float]],
        top_k: int = 10,
        fusion_method: str = "rrf",
        doc_id_mapping: Optional[Dict[int, str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Perform hybrid search combining keyword and semantic results.

        Args:
            query: Search query
            semantic_results: Results from semantic search
            top_k: Number of final results to return
            fusion_method: "rrf" or "weighted"
            doc_id_mapping: Mapping from doc indices to IDs

        Returns:
            Combined and reranked results
        """
        # Perform keyword search
        keyword_results = self.keyword_search(query, top_k * 2)

        # Merge results using specified method
        if fusion_method == "rrf":
            merged_results = self.reciprocal_rank_fusion(
                semantic_results, keyword_results, doc_id_mapping
            )
        else:
            merged_results = self.weighted_fusion(
                semantic_results, keyword_results, doc_id_mapping
            )

        # Return top-k results
        return merged_results[:top_k]

    def update_weights(
        self, semantic_weight: float, keyword_weight: float
    ) -> None:
        """
        Update the weights for hybrid search.

        Args:
            semantic_weight: New weight for semantic search
            keyword_weight: New weight for keyword search
        """
        # Normalize weights
        total = semantic_weight + keyword_weight
        self.semantic_weight = semantic_weight / total
        self.keyword_weight = keyword_weight / total

        logger.info(
            f"Updated weights - Semantic: {self.semantic_weight:.2f}, "
            f"Keyword: {self.keyword_weight:.2f}"
        )


class AdaptiveHybridSearch(HybridSearch):
    """
    Adaptive hybrid search that adjusts weights based on query characteristics.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.query_history = []
        self.performance_history = []

    def analyze_query(self, query: str) -> Dict[str, float]:
        """
        Analyze query characteristics to determine optimal weights.

        Args:
            query: Search query

        Returns:
            Dictionary with query characteristics
        """
        words = query.lower().split()

        characteristics = {
            "length": len(words),
            "has_technical_terms": any(
                word in ["api", "function", "class", "method", "error"]
                for word in words
            ),
            "has_question_words": any(
                word in ["what", "how", "why", "when", "where", "who"]
                for word in words
            ),
            "specificity": len(set(words)) / max(len(words), 1),
        }

        return characteristics

    def adaptive_weights(self, query: str) -> Tuple[float, float]:
        """
        Calculate adaptive weights based on query characteristics.

        Args:
            query: Search query

        Returns:
            Tuple of (semantic_weight, keyword_weight)
        """
        chars = self.analyze_query(query)

        # Short, specific queries benefit from keyword search
        if chars["length"] <= 3 and chars["specificity"] > 0.8:
            return 0.3, 0.7

        # Technical queries benefit from keyword search
        if chars["has_technical_terms"]:
            return 0.4, 0.6

        # Question queries benefit from semantic search
        if chars["has_question_words"]:
            return 0.7, 0.3

        # Default balanced weights
        return 0.5, 0.5

    def adaptive_hybrid_search(
        self,
        query: str,
        semantic_results: List[Tuple[str, float]],
        top_k: int = 10,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """
        Perform adaptive hybrid search with dynamic weight adjustment.

        Args:
            query: Search query
            semantic_results: Results from semantic search
            top_k: Number of final results to return
            **kwargs: Additional arguments for hybrid_search

        Returns:
            Combined and reranked results
        """
        # Get adaptive weights
        sem_weight, key_weight = self.adaptive_weights(query)
        self.update_weights(sem_weight, key_weight)

        # Perform hybrid search with adaptive weights
        results = self.hybrid_search(
            query, semantic_results, top_k, **kwargs
        )

        # Store query for learning
        self.query_history.append({
            "query": query,
            "weights": (sem_weight, key_weight),
            "num_results": len(results),
        })

        return results