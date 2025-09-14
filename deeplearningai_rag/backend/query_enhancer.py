"""
Query Enhancement module for advanced RAG techniques.
Implements query rewriting, decomposition, expansion, and HyDE.
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QueryEnhancer:
    """
    Enhances queries using various techniques to improve retrieval performance.
    """

    def __init__(self, ai_generator=None):
        """
        Initialize the query enhancer.

        Args:
            ai_generator: AI generator instance for query rewriting
        """
        self.ai_generator = ai_generator

    def rewrite_query(self, query: str, context: Optional[str] = None) -> List[str]:
        """
        Generate multiple query variations for better coverage.

        Args:
            query: Original query
            context: Optional context from conversation

        Returns:
            List of rewritten queries
        """
        if not self.ai_generator:
            return [query]

        prompt = f"""Rewrite the following query in 3 different ways to improve search results.
Keep the meaning the same but vary the phrasing and keywords.

Original query: {query}
{f"Context: {context}" if context else ""}

Provide exactly 3 variations, one per line:"""

        try:
            response = self.ai_generator.generate(prompt, max_tokens=200)
            variations = [line.strip() for line in response.strip().split("\n") if line.strip()]

            # Include original query
            variations.insert(0, query)

            # Limit to 4 queries total
            return variations[:4]
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return [query]

    def decompose_query(self, query: str) -> List[str]:
        """
        Break complex queries into sub-queries.

        Args:
            query: Complex query to decompose

        Returns:
            List of sub-queries
        """
        if not self.ai_generator:
            return [query]

        prompt = f"""Break down this complex query into simpler sub-queries.
Each sub-query should focus on one aspect of the original query.

Query: {query}

Provide sub-queries, one per line (maximum 3):"""

        try:
            response = self.ai_generator.generate(prompt, max_tokens=200)
            sub_queries = [line.strip() for line in response.strip().split("\n") if line.strip()]

            # Limit sub-queries
            return sub_queries[:3] if sub_queries else [query]
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return [query]

    def generate_hyde(self, query: str) -> str:
        """
        Generate a Hypothetical Document Embedding (HyDE).
        Creates a hypothetical answer to improve semantic search.

        Args:
            query: Original query

        Returns:
            Hypothetical document/answer
        """
        if not self.ai_generator:
            return ""

        prompt = f"""Write a brief, factual answer to this question that would appear in course materials.
Be specific and include relevant technical terms.

Question: {query}

Hypothetical answer (2-3 sentences):"""

        try:
            hyde_doc = self.ai_generator.generate(prompt, max_tokens=150)
            return hyde_doc.strip()
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            return ""

    def expand_query(self, query: str) -> str:
        """
        Expand query with synonyms and related terms.

        Args:
            query: Original query

        Returns:
            Expanded query
        """
        # Simple keyword expansion based on common patterns
        expansions = {
            "create": "create make build construct implement",
            "error": "error exception bug issue problem",
            "function": "function method procedure routine",
            "class": "class object type structure",
            "api": "api interface endpoint service",
            "database": "database db storage persistence",
            "model": "model algorithm network architecture",
            "train": "train training fit fitting learn learning",
            "test": "test testing evaluate validation",
            "deploy": "deploy deployment production release",
        }

        expanded = query.lower()
        for term, expansion in expansions.items():
            if term in expanded:
                expanded = expanded.replace(term, f"({expansion})")

        return expanded if expanded != query.lower() else query

    def enhance_query_pipeline(
        self,
        query: str,
        use_rewriting: bool = True,
        use_decomposition: bool = False,
        use_hyde: bool = True,
        use_expansion: bool = False,
        context: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Run the full query enhancement pipeline.

        Args:
            query: Original query
            use_rewriting: Whether to use query rewriting
            use_decomposition: Whether to decompose complex queries
            use_hyde: Whether to generate HyDE
            use_expansion: Whether to expand with synonyms
            context: Optional conversation context

        Returns:
            Dictionary with enhanced query components
        """
        enhanced = {
            "original": query,
            "variations": [],
            "sub_queries": [],
            "hyde": "",
            "expanded": query,
        }

        # Query rewriting
        if use_rewriting:
            enhanced["variations"] = self.rewrite_query(query, context)
            logger.info(f"Generated {len(enhanced['variations'])} query variations")

        # Query decomposition
        if use_decomposition and self._is_complex_query(query):
            enhanced["sub_queries"] = self.decompose_query(query)
            logger.info(f"Decomposed into {len(enhanced['sub_queries'])} sub-queries")

        # HyDE generation
        if use_hyde:
            enhanced["hyde"] = self.generate_hyde(query)
            if enhanced["hyde"]:
                logger.info("Generated HyDE document")

        # Query expansion
        if use_expansion:
            enhanced["expanded"] = self.expand_query(query)
            if enhanced["expanded"] != query:
                logger.info("Expanded query with synonyms")

        return enhanced

    def _is_complex_query(self, query: str) -> bool:
        """
        Determine if a query is complex enough to benefit from decomposition.

        Args:
            query: Query to analyze

        Returns:
            True if query is complex
        """
        # Simple heuristics for complexity
        words = query.lower().split()

        # Check for multiple concepts
        has_and = "and" in words
        has_or = "or" in words
        has_comparison = any(word in words for word in ["compare", "difference", "versus", "vs"])
        has_multiple_questions = query.count("?") > 1
        is_long = len(words) > 15

        return any([has_and, has_or, has_comparison, has_multiple_questions, is_long])


class SmartQueryRouter:
    """
    Routes queries to appropriate enhancement strategies based on query type.
    """

    def __init__(self, query_enhancer: QueryEnhancer):
        """
        Initialize the smart query router.

        Args:
            query_enhancer: QueryEnhancer instance
        """
        self.enhancer = query_enhancer

    def analyze_query_type(self, query: str) -> str:
        """
        Analyze and classify the query type.

        Args:
            query: Query to analyze

        Returns:
            Query type classification
        """
        query_lower = query.lower()

        # Classification rules
        if any(word in query_lower for word in ["what is", "define", "explain"]):
            return "definition"
        elif any(word in query_lower for word in ["how to", "how do", "steps"]):
            return "procedural"
        elif any(word in query_lower for word in ["why", "reason", "cause"]):
            return "conceptual"
        elif any(word in query_lower for word in ["compare", "difference", "versus"]):
            return "comparison"
        elif any(word in query_lower for word in ["error", "debug", "fix", "issue"]):
            return "troubleshooting"
        elif any(word in query_lower for word in ["example", "sample", "demonstrate"]):
            return "example"
        else:
            return "general"

    def route_query(self, query: str, context: Optional[str] = None) -> Dict[str, any]:
        """
        Route query to appropriate enhancement strategy.

        Args:
            query: Original query
            context: Optional conversation context

        Returns:
            Enhanced query components based on query type
        """
        query_type = self.analyze_query_type(query)
        logger.info(f"Query type identified: {query_type}")

        # Define enhancement strategies per query type
        strategies = {
            "definition": {
                "use_rewriting": True,
                "use_hyde": True,
                "use_decomposition": False,
                "use_expansion": True,
            },
            "procedural": {
                "use_rewriting": True,
                "use_hyde": False,
                "use_decomposition": True,
                "use_expansion": False,
            },
            "conceptual": {
                "use_rewriting": True,
                "use_hyde": True,
                "use_decomposition": False,
                "use_expansion": False,
            },
            "comparison": {
                "use_rewriting": True,
                "use_hyde": False,
                "use_decomposition": True,
                "use_expansion": False,
            },
            "troubleshooting": {
                "use_rewriting": True,
                "use_hyde": False,
                "use_decomposition": False,
                "use_expansion": True,
            },
            "example": {
                "use_rewriting": True,
                "use_hyde": True,
                "use_decomposition": False,
                "use_expansion": False,
            },
            "general": {
                "use_rewriting": True,
                "use_hyde": True,
                "use_decomposition": False,
                "use_expansion": False,
            },
        }

        strategy = strategies.get(query_type, strategies["general"])

        # Apply enhancement strategy
        enhanced = self.enhancer.enhance_query_pipeline(
            query,
            context=context,
            **strategy
        )

        enhanced["query_type"] = query_type
        return enhanced