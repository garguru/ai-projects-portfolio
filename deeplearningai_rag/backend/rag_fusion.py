"""
RAG-Fusion module implementing multi-query generation and rank fusion.
Advanced RAG technique for improved retrieval through query diversification.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class QueryFusion:
    """
    Generates multiple query variations and fuses results for better coverage.
    """

    def __init__(self, ai_generator=None, vector_store=None):
        """
        Initialize Query Fusion system.

        Args:
            ai_generator: AI generator for query variations
            vector_store: Vector store for retrieval
        """
        self.ai_generator = ai_generator
        self.vector_store = vector_store

    def generate_query_variations(
        self,
        original_query: str,
        num_variations: int = 3
    ) -> List[str]:
        """
        Generate multiple query variations for comprehensive search.

        Args:
            original_query: Original user query
            num_variations: Number of variations to generate

        Returns:
            List of query variations including the original
        """
        if not self.ai_generator:
            return [original_query]

        prompt = f"""Generate {num_variations} different ways to ask the same question.
Each variation should approach the topic from a slightly different angle while maintaining the core intent.

Original question: {original_query}

Generate exactly {num_variations} variations, each on a separate line:"""

        try:
            response = self.ai_generator.generate(prompt, max_tokens=200)
            variations = [line.strip() for line in response.strip().split("\n") if line.strip()]

            # Ensure we have the requested number of variations
            variations = variations[:num_variations]

            # Always include the original query first
            all_queries = [original_query] + variations
            return list(dict.fromkeys(all_queries))  # Remove duplicates while preserving order

        except Exception as e:
            logger.error(f"Failed to generate query variations: {e}")
            return [original_query]

    def parallel_search(
        self,
        queries: List[str],
        max_results_per_query: int = 10
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Perform parallel searches for all query variations.

        Args:
            queries: List of query variations
            max_results_per_query: Maximum results per query

        Returns:
            Dictionary mapping queries to their results
        """
        if not self.vector_store:
            logger.error("Vector store not available for parallel search")
            return {}

        results = {}
        for query in queries:
            try:
                # Perform semantic search
                search_results = self.vector_store.search_course_content(
                    query, max_results_per_query
                )
                results[query] = search_results
                logger.info(f"Query '{query}' returned {len(search_results)} results")
            except Exception as e:
                logger.error(f"Search failed for query '{query}': {e}")
                results[query] = []

        return results

    def reciprocal_rank_fusion(
        self,
        query_results: Dict[str, List[Tuple[str, float]]],
        k: int = 60
    ) -> List[Tuple[str, float]]:
        """
        Fuse results from multiple queries using Reciprocal Rank Fusion.

        Args:
            query_results: Dictionary mapping queries to their results
            k: RRF parameter (typically 60)

        Returns:
            Fused and reranked results
        """
        rrf_scores = {}
        doc_contents = {}

        for query, results in query_results.items():
            for rank, (doc_id, content, metadata, score) in enumerate(results):
                # Store document content for later retrieval
                doc_contents[doc_id] = (content, metadata)

                # Calculate RRF score
                rrf_score = 1.0 / (k + rank + 1)

                if doc_id in rrf_scores:
                    rrf_scores[doc_id] += rrf_score
                else:
                    rrf_scores[doc_id] = rrf_score

        # Sort by RRF score and prepare final results
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        for doc_id, rrf_score in sorted_docs:
            if doc_id in doc_contents:
                content, metadata = doc_contents[doc_id]
                final_results.append((doc_id, content, metadata, rrf_score))

        return final_results

    def weighted_fusion(
        self,
        query_results: Dict[str, List[Tuple[str, float]]],
        query_weights: Optional[Dict[str, float]] = None
    ) -> List[Tuple[str, float]]:
        """
        Fuse results using weighted score combination.

        Args:
            query_results: Dictionary mapping queries to their results
            query_weights: Optional weights for each query

        Returns:
            Fused and reranked results
        """
        if query_weights is None:
            # Equal weights for all queries
            query_weights = {q: 1.0 for q in query_results.keys()}

        # Normalize weights
        total_weight = sum(query_weights.values())
        normalized_weights = {q: w / total_weight for q, w in query_weights.items()}

        combined_scores = {}
        doc_contents = {}

        for query, results in query_results.items():
            weight = normalized_weights.get(query, 0.0)

            # Normalize scores within this query's results
            if results:
                scores = [score for _, _, _, score in results]
                max_score = max(scores) if scores else 1.0
                min_score = min(scores) if scores else 0.0
                score_range = max_score - min_score if max_score != min_score else 1.0

                for doc_id, content, metadata, score in results:
                    # Store document content
                    doc_contents[doc_id] = (content, metadata)

                    # Normalize and weight the score
                    normalized_score = (score - min_score) / score_range
                    weighted_score = normalized_score * weight

                    if doc_id in combined_scores:
                        combined_scores[doc_id] += weighted_score
                    else:
                        combined_scores[doc_id] = weighted_score

        # Sort and prepare final results
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        for doc_id, final_score in sorted_docs:
            if doc_id in doc_contents:
                content, metadata = doc_contents[doc_id]
                final_results.append((doc_id, content, metadata, final_score))

        return final_results

    def diversity_fusion(
        self,
        query_results: Dict[str, List[Tuple[str, float]]],
        diversity_weight: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Fuse results with diversity consideration.

        Args:
            query_results: Dictionary mapping queries to their results
            diversity_weight: Weight for diversity vs relevance

        Returns:
            Fused results with diversity consideration
        """
        # First, get initial fusion using RRF
        initial_results = self.reciprocal_rank_fusion(query_results)

        if len(initial_results) <= 1:
            return initial_results

        # Apply diversity selection
        selected_results = []
        remaining_results = initial_results.copy()

        while remaining_results and len(selected_results) < len(initial_results):
            best_score = float('-inf')
            best_idx = 0

            for i, (doc_id, content, metadata, score) in enumerate(remaining_results):
                # Calculate diversity penalty
                diversity_penalty = 0.0
                if selected_results:
                    # Simple diversity based on content similarity
                    current_words = set(content.lower().split())
                    for _, selected_content, _, _ in selected_results:
                        selected_words = set(selected_content.lower().split())
                        overlap = len(current_words.intersection(selected_words))
                        similarity = overlap / max(len(current_words), len(selected_words), 1)
                        diversity_penalty = max(diversity_penalty, similarity)

                # Combined score with diversity
                combined_score = score * (1 - diversity_weight) - diversity_penalty * diversity_weight

                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = i

            # Move the best document to selected
            selected_results.append(remaining_results.pop(best_idx))

        return selected_results

    def fusion_search(
        self,
        original_query: str,
        num_variations: int = 3,
        max_results_per_query: int = 10,
        fusion_method: str = "rrf",
        final_top_k: int = 10,
        **kwargs
    ) -> List[Tuple[str, str, Dict, float]]:
        """
        Complete RAG-Fusion search pipeline.

        Args:
            original_query: Original user query
            num_variations: Number of query variations to generate
            max_results_per_query: Max results per individual query
            fusion_method: "rrf", "weighted", or "diversity"
            final_top_k: Final number of results to return
            **kwargs: Additional arguments for specific fusion methods

        Returns:
            Fused and reranked search results
        """
        logger.info(f"Starting RAG-Fusion search for: '{original_query}'")

        # Step 1: Generate query variations
        query_variations = self.generate_query_variations(original_query, num_variations)
        logger.info(f"Generated {len(query_variations)} query variations")

        # Step 2: Perform parallel searches
        query_results = self.parallel_search(query_variations, max_results_per_query)

        # Step 3: Fuse results
        if fusion_method == "rrf":
            fused_results = self.reciprocal_rank_fusion(
                query_results, k=kwargs.get("rrf_k", 60)
            )
        elif fusion_method == "weighted":
            fused_results = self.weighted_fusion(
                query_results, kwargs.get("query_weights")
            )
        elif fusion_method == "diversity":
            fused_results = self.diversity_fusion(
                query_results, kwargs.get("diversity_weight", 0.3)
            )
        else:
            logger.warning(f"Unknown fusion method: {fusion_method}. Using RRF.")
            fused_results = self.reciprocal_rank_fusion(query_results)

        logger.info(f"Fusion method '{fusion_method}' produced {len(fused_results)} results")

        # Step 4: Return top-k results
        final_results = fused_results[:final_top_k]
        logger.info(f"Returning top {len(final_results)} fused results")

        return final_results


class AdaptiveRAGFusion(QueryFusion):
    """
    Adaptive RAG-Fusion that learns from query patterns and user feedback.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_patterns = {}
        self.performance_history = []

    def analyze_query_complexity(self, query: str) -> Dict[str, any]:
        """
        Analyze query complexity to determine optimal fusion parameters.

        Args:
            query: Query to analyze

        Returns:
            Dictionary with complexity metrics
        """
        words = query.lower().split()

        analysis = {
            "word_count": len(words),
            "unique_words": len(set(words)),
            "has_technical_terms": any(
                term in query.lower()
                for term in ["api", "function", "class", "error", "debug"]
            ),
            "has_question_words": any(
                word in words for word in ["what", "how", "why", "when", "where"]
            ),
            "is_comparative": any(
                word in words for word in ["compare", "difference", "versus", "vs"]
            ),
            "complexity_score": len(words) * len(set(words)) / max(len(words), 1)
        }

        return analysis

    def adaptive_fusion_search(
        self,
        original_query: str,
        **kwargs
    ) -> List[Tuple[str, str, Dict, float]]:
        """
        Perform adaptive fusion search with optimized parameters.

        Args:
            original_query: Original user query
            **kwargs: Override parameters

        Returns:
            Adaptively fused search results
        """
        # Analyze query
        analysis = self.analyze_query_complexity(original_query)
        logger.info(f"Query complexity analysis: {analysis}")

        # Adaptive parameter selection
        if analysis["complexity_score"] > 2.0:
            # Complex query - use more variations and diversity
            num_variations = kwargs.get("num_variations", 4)
            fusion_method = kwargs.get("fusion_method", "diversity")
            diversity_weight = kwargs.get("diversity_weight", 0.4)
        elif analysis["has_technical_terms"]:
            # Technical query - use weighted fusion with preference for precision
            num_variations = kwargs.get("num_variations", 2)
            fusion_method = kwargs.get("fusion_method", "weighted")
        else:
            # Simple query - use standard RRF
            num_variations = kwargs.get("num_variations", 3)
            fusion_method = kwargs.get("fusion_method", "rrf")

        # Perform fusion search with adaptive parameters
        results = self.fusion_search(
            original_query,
            num_variations=num_variations,
            fusion_method=fusion_method,
            **kwargs
        )

        # Store pattern for future learning
        self.query_patterns[original_query] = {
            "analysis": analysis,
            "parameters": {
                "num_variations": num_variations,
                "fusion_method": fusion_method,
            },
            "result_count": len(results)
        }

        return results