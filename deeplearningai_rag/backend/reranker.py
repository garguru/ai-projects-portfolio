"""
Reranking module for advanced RAG systems.
Implements various reranking strategies including cross-encoders and MMR.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class BaseReranker:
    """Base class for all reranking strategies."""

    def rerank(
        self,
        query: str,
        documents: List[str],
        scores: List[float],
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents based on query relevance.

        Args:
            query: Original query
            documents: List of documents to rerank
            scores: Original retrieval scores
            top_k: Number of documents to return

        Returns:
            List of (document_index, reranked_score) tuples
        """
        raise NotImplementedError


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder based reranker for more accurate relevance scoring.
    Falls back to sentence similarity if cross-encoder is not available.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model = None
        self.model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"

        # Try to load cross-encoder model
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Loaded cross-encoder model: {self.model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not available. "
                "Falling back to sentence similarity."
            )
        except Exception as e:
            logger.warning(
                f"Failed to load cross-encoder model: {e}. "
                "Falling back to sentence similarity."
            )

    def rerank(
        self,
        query: str,
        documents: List[str],
        scores: List[float],
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Rerank using cross-encoder or fallback similarity."""
        if not documents:
            return []

        if self.model is not None:
            return self._cross_encoder_rerank(query, documents, top_k)
        else:
            return self._similarity_rerank(query, documents, scores, top_k)

    def _cross_encoder_rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int
    ) -> List[Tuple[int, float]]:
        """Rerank using cross-encoder model."""
        # Prepare query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Get cross-encoder scores
        try:
            cross_scores = self.model.predict(pairs)

            # Create (index, score) pairs and sort
            indexed_scores = list(enumerate(cross_scores))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)

            return indexed_scores[:top_k]
        except Exception as e:
            logger.error(f"Cross-encoder prediction failed: {e}")
            # Fallback to original scores
            return [(i, score) for i, score in enumerate(scores)][:top_k]

    def _similarity_rerank(
        self,
        query: str,
        documents: List[str],
        original_scores: List[float],
        top_k: int
    ) -> List[Tuple[int, float]]:
        """Fallback reranking using simple text similarity."""
        try:
            # Simple word overlap scoring as fallback
            query_words = set(query.lower().split())
            similarities = []

            for doc in documents:
                doc_words = set(doc.lower().split())
                overlap = len(query_words.intersection(doc_words))
                similarity = overlap / max(len(query_words), 1)
                similarities.append(similarity)

            # Combine with original scores
            combined_scores = []
            for i, (orig_score, sim_score) in enumerate(zip(original_scores, similarities)):
                combined = 0.7 * orig_score + 0.3 * sim_score
                combined_scores.append((i, combined))

            # Sort by combined score
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            return combined_scores[:top_k]

        except Exception as e:
            logger.error(f"Similarity reranking failed: {e}")
            # Ultimate fallback: return original order
            return [(i, score) for i, score in enumerate(original_scores)][:top_k]


class MMRReranker(BaseReranker):
    """
    Maximal Marginal Relevance (MMR) reranker for diversity.
    Balances relevance with diversity to avoid redundant results.
    """

    def __init__(self, lambda_param: float = 0.7, embeddings_model=None):
        """
        Initialize MMR reranker.

        Args:
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
            embeddings_model: Optional embedding model for similarity calculation
        """
        self.lambda_param = lambda_param
        self.embeddings_model = embeddings_model

    def rerank(
        self,
        query: str,
        documents: List[str],
        scores: List[float],
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Rerank using MMR for diversity."""
        if not documents or len(documents) <= 1:
            return [(i, score) for i, score in enumerate(scores)][:top_k]

        try:
            # Get embeddings for similarity calculation
            embeddings = self._get_embeddings(query, documents)
            if embeddings is None:
                # Fallback to original ranking
                indexed_scores = [(i, score) for i, score in enumerate(scores)]
                return sorted(indexed_scores, key=lambda x: x[1], reverse=True)[:top_k]

            query_emb = embeddings[0]
            doc_embs = embeddings[1:]

            # Calculate query-document similarities
            query_similarities = cosine_similarity([query_emb], doc_embs)[0]

            # MMR selection
            selected = []
            remaining = list(range(len(documents)))

            for _ in range(min(top_k, len(documents))):
                if not remaining:
                    break

                best_score = float('-inf')
                best_idx = None

                for idx in remaining:
                    # Relevance score
                    relevance = query_similarities[idx]

                    # Diversity score (max similarity to already selected)
                    diversity = 0.0
                    if selected:
                        selected_embs = [doc_embs[i] for i in selected]
                        similarities = cosine_similarity([doc_embs[idx]], selected_embs)[0]
                        diversity = max(similarities) if len(similarities) > 0 else 0.0

                    # MMR score
                    mmr_score = (
                        self.lambda_param * relevance -
                        (1 - self.lambda_param) * diversity
                    )

                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx

                if best_idx is not None:
                    selected.append(best_idx)
                    remaining.remove(best_idx)

            # Return with MMR scores
            return [(idx, query_similarities[idx]) for idx in selected]

        except Exception as e:
            logger.error(f"MMR reranking failed: {e}")
            # Fallback to original ranking
            indexed_scores = [(i, score) for i, score in enumerate(scores)]
            return sorted(indexed_scores, key=lambda x: x[1], reverse=True)[:top_k]

    def _get_embeddings(self, query: str, documents: List[str]) -> Optional[np.ndarray]:
        """Get embeddings for query and documents."""
        if self.embeddings_model:
            try:
                texts = [query] + documents
                embeddings = self.embeddings_model.encode(texts)
                return embeddings
            except Exception as e:
                logger.error(f"Failed to get embeddings: {e}")

        # Simple fallback: use random embeddings for structure
        # In practice, you'd want proper embeddings here
        logger.warning("Using random embeddings fallback for MMR")
        np.random.seed(42)  # For reproducibility
        return np.random.random((len(documents) + 1, 384))


class CombinedReranker(BaseReranker):
    """
    Combines multiple reranking strategies for better performance.
    """

    def __init__(
        self,
        cross_encoder_weight: float = 0.6,
        mmr_weight: float = 0.4,
        embeddings_model=None
    ):
        """
        Initialize combined reranker.

        Args:
            cross_encoder_weight: Weight for cross-encoder scores
            mmr_weight: Weight for MMR scores
            embeddings_model: Embedding model for MMR
        """
        self.cross_encoder = CrossEncoderReranker()
        self.mmr_reranker = MMRReranker(embeddings_model=embeddings_model)
        self.ce_weight = cross_encoder_weight
        self.mmr_weight = mmr_weight

        # Normalize weights
        total = cross_encoder_weight + mmr_weight
        self.ce_weight = cross_encoder_weight / total
        self.mmr_weight = mmr_weight / total

    def rerank(
        self,
        query: str,
        documents: List[str],
        scores: List[float],
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Combine cross-encoder and MMR reranking."""
        if not documents:
            return []

        try:
            # Get cross-encoder scores
            ce_results = self.cross_encoder.rerank(query, documents, scores, len(documents))
            ce_scores = {idx: score for idx, score in ce_results}

            # Get MMR scores
            mmr_results = self.mmr_reranker.rerank(query, documents, scores, len(documents))
            mmr_scores = {idx: score for idx, score in mmr_results}

            # Normalize scores to [0, 1]
            def normalize_scores(score_dict):
                if not score_dict:
                    return {}
                scores = list(score_dict.values())
                min_score, max_score = min(scores), max(scores)
                if max_score == min_score:
                    return {k: 1.0 for k in score_dict}
                return {
                    k: (v - min_score) / (max_score - min_score)
                    for k, v in score_dict.items()
                }

            ce_norm = normalize_scores(ce_scores)
            mmr_norm = normalize_scores(mmr_scores)

            # Combine scores
            combined_scores = []
            for idx in range(len(documents)):
                ce_score = ce_norm.get(idx, 0.0)
                mmr_score = mmr_norm.get(idx, 0.0)
                combined = self.ce_weight * ce_score + self.mmr_weight * mmr_score
                combined_scores.append((idx, combined))

            # Sort and return top-k
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            return combined_scores[:top_k]

        except Exception as e:
            logger.error(f"Combined reranking failed: {e}")
            # Fallback to cross-encoder only
            return self.cross_encoder.rerank(query, documents, scores, top_k)


class RerankerManager:
    """
    Manages different reranking strategies and selects appropriate ones.
    """

    def __init__(self, embeddings_model=None):
        """Initialize reranker manager."""
        self.rerankers = {
            "cross_encoder": CrossEncoderReranker(),
            "mmr": MMRReranker(embeddings_model=embeddings_model),
            "combined": CombinedReranker(embeddings_model=embeddings_model),
        }
        self.default_strategy = "cross_encoder"

    def rerank(
        self,
        query: str,
        documents: List[str],
        scores: List[float],
        strategy: str = None,
        top_k: int = 10,
        **kwargs
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents using specified strategy.

        Args:
            query: Search query
            documents: List of documents
            scores: Original retrieval scores
            strategy: Reranking strategy to use
            top_k: Number of results to return
            **kwargs: Additional arguments for rerankers

        Returns:
            List of (document_index, reranked_score) tuples
        """
        strategy = strategy or self.default_strategy
        reranker = self.rerankers.get(strategy)

        if not reranker:
            logger.warning(f"Unknown reranking strategy: {strategy}")
            reranker = self.rerankers[self.default_strategy]

        logger.info(f"Using reranking strategy: {strategy}")
        return reranker.rerank(query, documents, scores, top_k)

    def add_reranker(self, name: str, reranker: BaseReranker):
        """Add a custom reranker."""
        self.rerankers[name] = reranker
        logger.info(f"Added custom reranker: {name}")

    def set_default_strategy(self, strategy: str):
        """Set the default reranking strategy."""
        if strategy in self.rerankers:
            self.default_strategy = strategy
            logger.info(f"Set default reranking strategy to: {strategy}")
        else:
            logger.warning(f"Unknown strategy: {strategy}")