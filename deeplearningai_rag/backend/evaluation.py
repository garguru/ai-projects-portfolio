"""
Evaluation module for RAG system performance metrics and analysis.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a single query execution"""
    query: str
    session_id: str
    timestamp: str

    # Performance metrics
    total_time: float  # Total query processing time
    retrieval_time: float  # Time for document retrieval
    generation_time: float  # Time for answer generation

    # Retrieval metrics
    num_results_retrieved: int  # Number of documents retrieved
    retrieval_method: str  # "semantic", "hybrid", "rag_fusion", etc.

    # Quality indicators
    answer_length: int  # Length of generated answer in characters
    sources_count: int  # Number of source documents cited

    # Enhanced metrics
    used_query_enhancement: bool = False
    used_reranking: bool = False
    used_hyde: bool = False
    fusion_method: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class SystemMetrics:
    """Aggregated system performance metrics"""
    total_queries: int = 0
    avg_query_time: float = 0.0
    avg_retrieval_time: float = 0.0
    avg_generation_time: float = 0.0
    avg_results_per_query: float = 0.0
    avg_answer_length: float = 0.0

    # Method usage statistics
    semantic_search_count: int = 0
    hybrid_search_count: int = 0
    rag_fusion_count: int = 0
    query_enhancement_count: int = 0
    reranking_count: int = 0

    # Performance distribution
    fast_queries: int = 0  # < 2 seconds
    medium_queries: int = 0  # 2-5 seconds
    slow_queries: int = 0  # > 5 seconds


class RAGEvaluator:
    """Evaluates and tracks RAG system performance"""

    def __init__(self, log_file_path: str = "rag_metrics.jsonl"):
        self.log_file_path = log_file_path
        self.query_metrics: List[QueryMetrics] = []
        self.system_metrics = SystemMetrics()

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file_path) if os.path.dirname(log_file_path) else ".", exist_ok=True)

    def start_query_measurement(self, query: str, session_id: str) -> Dict[str, Any]:
        """Start measuring a query. Returns context for tracking"""
        return {
            "query": query,
            "session_id": session_id,
            "start_time": time.time(),
            "retrieval_start": None,
            "retrieval_end": None,
            "generation_start": None,
            "generation_end": None,
            "retrieval_method": "unknown",
            "num_results": 0,
            "sources_count": 0,
            "answer_length": 0,
            "used_query_enhancement": False,
            "used_reranking": False,
            "used_hyde": False,
            "fusion_method": None,
        }

    def mark_retrieval_start(self, context: Dict[str, Any]):
        """Mark the start of retrieval phase"""
        context["retrieval_start"] = time.time()

    def mark_retrieval_end(
        self,
        context: Dict[str, Any],
        num_results: int,
        method: str = "semantic"
    ):
        """Mark the end of retrieval phase"""
        context["retrieval_end"] = time.time()
        context["num_results"] = num_results
        context["retrieval_method"] = method

    def mark_generation_start(self, context: Dict[str, Any]):
        """Mark the start of answer generation"""
        context["generation_start"] = time.time()

    def mark_generation_end(
        self,
        context: Dict[str, Any],
        answer: str,
        sources_count: int
    ):
        """Mark the end of answer generation"""
        context["generation_end"] = time.time()
        context["answer_length"] = len(answer)
        context["sources_count"] = sources_count

    def set_enhancement_flags(
        self,
        context: Dict[str, Any],
        used_query_enhancement: bool = False,
        used_reranking: bool = False,
        used_hyde: bool = False,
        fusion_method: Optional[str] = None
    ):
        """Set flags for advanced features used"""
        context["used_query_enhancement"] = used_query_enhancement
        context["used_reranking"] = used_reranking
        context["used_hyde"] = used_hyde
        context["fusion_method"] = fusion_method

    def finish_query_measurement(self, context: Dict[str, Any]) -> QueryMetrics:
        """Finish measuring a query and return metrics"""
        end_time = time.time()

        # Calculate times
        total_time = end_time - context["start_time"]

        retrieval_time = 0.0
        if context["retrieval_start"] and context["retrieval_end"]:
            retrieval_time = context["retrieval_end"] - context["retrieval_start"]

        generation_time = 0.0
        if context["generation_start"] and context["generation_end"]:
            generation_time = context["generation_end"] - context["generation_start"]

        # Create metrics object
        metrics = QueryMetrics(
            query=context["query"],
            session_id=context["session_id"],
            timestamp=datetime.now().isoformat(),
            total_time=total_time,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            num_results_retrieved=context["num_results"],
            retrieval_method=context["retrieval_method"],
            answer_length=context["answer_length"],
            sources_count=context["sources_count"],
            used_query_enhancement=context["used_query_enhancement"],
            used_reranking=context["used_reranking"],
            used_hyde=context["used_hyde"],
            fusion_method=context["fusion_method"],
        )

        # Store and log metrics
        self.query_metrics.append(metrics)
        self._log_metrics(metrics)
        self._update_system_metrics(metrics)

        return metrics

    def _log_metrics(self, metrics: QueryMetrics):
        """Log metrics to JSONL file"""
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def _update_system_metrics(self, query_metrics: QueryMetrics):
        """Update aggregated system metrics"""
        self.system_metrics.total_queries += 1
        n = self.system_metrics.total_queries

        # Update running averages
        self.system_metrics.avg_query_time = (
            (self.system_metrics.avg_query_time * (n - 1) + query_metrics.total_time) / n
        )
        self.system_metrics.avg_retrieval_time = (
            (self.system_metrics.avg_retrieval_time * (n - 1) + query_metrics.retrieval_time) / n
        )
        self.system_metrics.avg_generation_time = (
            (self.system_metrics.avg_generation_time * (n - 1) + query_metrics.generation_time) / n
        )
        self.system_metrics.avg_results_per_query = (
            (self.system_metrics.avg_results_per_query * (n - 1) + query_metrics.num_results_retrieved) / n
        )
        self.system_metrics.avg_answer_length = (
            (self.system_metrics.avg_answer_length * (n - 1) + query_metrics.answer_length) / n
        )

        # Update method usage counts
        if query_metrics.retrieval_method == "semantic":
            self.system_metrics.semantic_search_count += 1
        elif query_metrics.retrieval_method == "hybrid":
            self.system_metrics.hybrid_search_count += 1
        elif query_metrics.retrieval_method == "rag_fusion":
            self.system_metrics.rag_fusion_count += 1

        if query_metrics.used_query_enhancement:
            self.system_metrics.query_enhancement_count += 1

        if query_metrics.used_reranking:
            self.system_metrics.reranking_count += 1

        # Update performance distribution
        if query_metrics.total_time < 2.0:
            self.system_metrics.fast_queries += 1
        elif query_metrics.total_time < 5.0:
            self.system_metrics.medium_queries += 1
        else:
            self.system_metrics.slow_queries += 1

    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        return self.system_metrics

    def get_recent_queries(self, limit: int = 10) -> List[QueryMetrics]:
        """Get most recent query metrics"""
        return self.query_metrics[-limit:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a comprehensive performance summary"""
        metrics = self.system_metrics

        if metrics.total_queries == 0:
            return {"error": "No queries processed yet"}

        return {
            "total_queries": metrics.total_queries,
            "average_times": {
                "total": f"{metrics.avg_query_time:.2f}s",
                "retrieval": f"{metrics.avg_retrieval_time:.2f}s",
                "generation": f"{metrics.avg_generation_time:.2f}s",
            },
            "retrieval_stats": {
                "avg_results_per_query": f"{metrics.avg_results_per_query:.1f}",
                "avg_answer_length": f"{metrics.avg_answer_length:.0f} chars",
            },
            "method_usage": {
                "semantic_search": f"{metrics.semantic_search_count} ({metrics.semantic_search_count/metrics.total_queries*100:.1f}%)",
                "hybrid_search": f"{metrics.hybrid_search_count} ({metrics.hybrid_search_count/metrics.total_queries*100:.1f}%)",
                "rag_fusion": f"{metrics.rag_fusion_count} ({metrics.rag_fusion_count/metrics.total_queries*100:.1f}%)",
            },
            "feature_usage": {
                "query_enhancement": f"{metrics.query_enhancement_count} ({metrics.query_enhancement_count/metrics.total_queries*100:.1f}%)",
                "reranking": f"{metrics.reranking_count} ({metrics.reranking_count/metrics.total_queries*100:.1f}%)",
            },
            "performance_distribution": {
                "fast_queries": f"{metrics.fast_queries} (<2s)",
                "medium_queries": f"{metrics.medium_queries} (2-5s)",
                "slow_queries": f"{metrics.slow_queries} (>5s)",
            },
        }

    def analyze_query_patterns(self) -> Dict[str, Any]:
        """Analyze query patterns and performance"""
        if not self.query_metrics:
            return {"error": "No query data available"}

        # Group by retrieval method
        method_performance = defaultdict(list)
        for query in self.query_metrics:
            method_performance[query.retrieval_method].append(query.total_time)

        method_avg_times = {}
        for method, times in method_performance.items():
            method_avg_times[method] = sum(times) / len(times)

        # Feature impact analysis
        enhanced_queries = [q for q in self.query_metrics if q.used_query_enhancement]
        regular_queries = [q for q in self.query_metrics if not q.used_query_enhancement]

        enhanced_avg_time = sum(q.total_time for q in enhanced_queries) / len(enhanced_queries) if enhanced_queries else 0
        regular_avg_time = sum(q.total_time for q in regular_queries) / len(regular_queries) if regular_queries else 0

        return {
            "total_queries": len(self.query_metrics),
            "method_performance": method_avg_times,
            "enhancement_impact": {
                "enhanced_queries_avg_time": f"{enhanced_avg_time:.2f}s",
                "regular_queries_avg_time": f"{regular_avg_time:.2f}s",
                "enhancement_overhead": f"{enhanced_avg_time - regular_avg_time:.2f}s",
            },
            "recent_trend": {
                "last_10_avg_time": f"{sum(q.total_time for q in self.query_metrics[-10:]) / min(10, len(self.query_metrics)):.2f}s",
            }
        }

    def export_metrics(self, output_file: str = "rag_analysis.json"):
        """Export comprehensive metrics analysis"""
        analysis = {
            "system_metrics": asdict(self.system_metrics),
            "performance_summary": self.get_performance_summary(),
            "query_patterns": self.analyze_query_patterns(),
            "export_timestamp": datetime.now().isoformat(),
            "recent_queries": [q.to_dict() for q in self.get_recent_queries(20)]
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        logger.info(f"Metrics exported to {output_file}")
        return analysis