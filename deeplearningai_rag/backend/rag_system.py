import os
import logging
from typing import Dict, List, Optional, Tuple, Any

from ai_generator import AIGenerator
from document_processor import DocumentProcessor
from models import Course, CourseChunk, Lesson
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from session_manager import SessionManager
from vector_store import VectorStore

# Import new advanced RAG components
from hybrid_search import HybridSearch, AdaptiveHybridSearch
from query_enhancer import QueryEnhancer, SmartQueryRouter
from reranker import RerankerManager
from rag_fusion import QueryFusion, AdaptiveRAGFusion
from evaluation import RAGEvaluator

logger = logging.getLogger(__name__)


class EnhancedRAGSystem:
    """Advanced Retrieval-Augmented Generation system with hybrid search, query enhancement, and reranking"""

    def __init__(self, config):
        self.config = config
        logger.info("Initializing Enhanced RAG System with advanced features")

        # Initialize core components
        self.document_processor = DocumentProcessor(
            config.CHUNK_SIZE, config.CHUNK_OVERLAP
        )
        self.vector_store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )
        self.ai_generator = AIGenerator(
            config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL
        )
        self.session_manager = SessionManager(config.MAX_HISTORY)

        # Initialize search tools (legacy support)
        self.tool_manager = ToolManager()
        self.search_tool = CourseSearchTool(self.vector_store)
        self.outline_tool = CourseOutlineTool(self.vector_store)
        self.tool_manager.register_tool(self.search_tool)
        self.tool_manager.register_tool(self.outline_tool)

        # Initialize advanced RAG components
        self._init_advanced_components()

        # Initialize evaluation system
        if config.ENABLE_METRICS:
            self.evaluator = RAGEvaluator("rag_metrics.jsonl")
            logger.info("Evaluation system enabled")
        else:
            self.evaluator = None

    def _init_advanced_components(self):
        """Initialize advanced RAG components based on configuration"""
        config = self.config

        # Hybrid Search
        if config.USE_HYBRID_SEARCH:
            self.hybrid_search = AdaptiveHybridSearch(
                semantic_weight=config.SEMANTIC_WEIGHT,
                keyword_weight=config.KEYWORD_WEIGHT,
                rrf_k=config.RRF_K
            )
            logger.info("Adaptive hybrid search enabled")
        else:
            self.hybrid_search = None

        # Query Enhancement
        if config.USE_QUERY_ENHANCEMENT:
            self.query_enhancer = QueryEnhancer(self.ai_generator)
            self.query_router = SmartQueryRouter(self.query_enhancer)
            logger.info("Query enhancement enabled")
        else:
            self.query_enhancer = None
            self.query_router = None

        # Reranking
        if config.USE_RERANKING:
            # Get embedding model for MMR if needed
            embeddings_model = None
            if config.RERANK_STRATEGY in ["mmr", "combined"]:
                try:
                    from sentence_transformers import SentenceTransformer
                    embeddings_model = SentenceTransformer(config.EMBEDDING_MODEL)
                except ImportError:
                    logger.warning("sentence-transformers not available for MMR reranking")

            self.reranker_manager = RerankerManager(embeddings_model)
            logger.info(f"Reranking enabled with strategy: {config.RERANK_STRATEGY}")
        else:
            self.reranker_manager = None

        # RAG-Fusion
        if config.USE_RAG_FUSION:
            self.rag_fusion = AdaptiveRAGFusion(
                self.ai_generator, self.vector_store
            )
            logger.info("RAG-Fusion enabled")
        else:
            self.rag_fusion = None

    def add_course_document(self, file_path: str) -> Tuple[Course, int]:
        """
        Add a single course document to the knowledge base.

        Args:
            file_path: Path to the course document

        Returns:
            Tuple of (Course object, number of chunks created)
        """
        try:
            # Process the document
            course, course_chunks = self.document_processor.process_course_document(
                file_path
            )

            # Add course metadata to vector store for semantic search
            self.vector_store.add_course_metadata(course)

            # Add course content chunks to vector store
            self.vector_store.add_course_content(course_chunks)

            # Rebuild BM25 index if hybrid search is enabled
            if self.config.USE_HYBRID_SEARCH and self.hybrid_search:
                self.vector_store.build_bm25_index()

            return course, len(course_chunks)
        except Exception as e:
            print(f"Error processing course document {file_path}: {e}")
            return None, 0

    def add_course_folder(
        self, folder_path: str, clear_existing: bool = False
    ) -> Tuple[int, int]:
        """
        Add all course documents from a folder.

        Args:
            folder_path: Path to folder containing course documents
            clear_existing: Whether to clear existing data first

        Returns:
            Tuple of (total courses added, total chunks created)
        """
        total_courses = 0
        total_chunks = 0

        # Clear existing data if requested
        if clear_existing:
            print("Clearing existing data for fresh rebuild...")
            self.vector_store.clear_all_data()

        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist")
            return 0, 0

        # Get existing course titles to avoid re-processing
        existing_course_titles = set(self.vector_store.get_existing_course_titles())

        # Process each file in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(
                (".pdf", ".docx", ".txt")
            ):
                try:
                    # Check if this course might already exist
                    # We'll process the document to get the course ID, but only add if new
                    course, course_chunks = (
                        self.document_processor.process_course_document(file_path)
                    )

                    if course and course.title not in existing_course_titles:
                        # This is a new course - add it to the vector store
                        self.vector_store.add_course_metadata(course)
                        self.vector_store.add_course_content(course_chunks)
                        total_courses += 1
                        total_chunks += len(course_chunks)
                        print(
                            f"Added new course: {course.title} ({len(course_chunks)} chunks)"
                        )
                        existing_course_titles.add(course.title)
                    elif course:
                        print(f"Course already exists: {course.title} - skipping")
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

        # Rebuild BM25 index after adding multiple courses
        if total_courses > 0 and self.config.USE_HYBRID_SEARCH and self.hybrid_search:
            print("Rebuilding BM25 index for hybrid search...")
            self.vector_store.build_bm25_index()

        return total_courses, total_chunks

    def query(
        self, query: str, session_id: Optional[str] = None
    ) -> Tuple[str, List[str], List[str]]:
        """
        Process a user query using the enhanced RAG system.

        Args:
            query: User's question
            session_id: Optional session ID for conversation context

        Returns:
            Tuple of (response, sources list, source_links list)
        """
        # Start evaluation tracking
        eval_context = None
        if self.evaluator:
            eval_context = self.evaluator.start_query_measurement(query, session_id or "anonymous")

        try:
            # Get conversation history
            history = None
            if session_id:
                history = self.session_manager.get_conversation_history(session_id)

            # Choose between advanced retrieval or fallback to legacy tools
            if self._should_use_advanced_retrieval():
                response, sources, source_links = self._advanced_query_pipeline(
                    query, history, eval_context
                )
            else:
                # Fallback to legacy tool-based approach
                response, sources, source_links = self._legacy_query_pipeline(
                    query, history, eval_context
                )

            # Update conversation history
            if session_id:
                self.session_manager.add_exchange(session_id, query, response)

            # Finish evaluation tracking
            if self.evaluator and eval_context:
                self.evaluator.finish_query_measurement(eval_context)

            return response, sources, source_links

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            if self.evaluator and eval_context:
                eval_context["error"] = str(e)
                self.evaluator.finish_query_measurement(eval_context)

            # Fallback to simple response
            return f"I encountered an error processing your query: {e}", [], []

    def _should_use_advanced_retrieval(self) -> bool:
        """Determine whether to use advanced retrieval features"""
        return (
            self.config.USE_HYBRID_SEARCH or
            self.config.USE_QUERY_ENHANCEMENT or
            self.config.USE_RAG_FUSION or
            self.config.USE_RERANKING
        )

    def _advanced_query_pipeline(
        self,
        query: str,
        history: Optional[List],
        eval_context: Optional[Dict]
    ) -> Tuple[str, List[str], List[str]]:
        """Advanced query processing pipeline with all enhanced features"""

        # Phase 1: Query Enhancement
        enhanced_query_data = self._enhance_query(query, history, eval_context)

        # Phase 2: Retrieval (Hybrid/RAG-Fusion)
        if eval_context:
            self.evaluator.mark_retrieval_start(eval_context)

        retrieval_results = self._perform_retrieval(enhanced_query_data, eval_context)

        if eval_context:
            self.evaluator.mark_retrieval_end(
                eval_context,
                len(retrieval_results),
                enhanced_query_data.get("retrieval_method", "advanced")
            )

        # Phase 3: Reranking
        reranked_results = self._apply_reranking(query, retrieval_results, eval_context)

        # Phase 4: Answer Generation
        if eval_context:
            self.evaluator.mark_generation_start(eval_context)

        response, sources, source_links = self._generate_answer(
            query, reranked_results, history
        )

        if eval_context:
            self.evaluator.mark_generation_end(eval_context, response, len(sources))

        return response, sources, source_links

    def _enhance_query(
        self,
        query: str,
        history: Optional[List],
        eval_context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Apply query enhancement techniques"""
        enhanced_data = {"original_query": query, "retrieval_method": "semantic"}

        if not self.config.USE_QUERY_ENHANCEMENT or not self.query_router:
            return enhanced_data

        try:
            # Get conversation context
            context = None
            if history and len(history) > 0:
                context = f"Previous conversation: {history[-1]}"

            # Apply smart query routing
            enhancement_result = self.query_router.route_query(query, context)

            enhanced_data.update({
                "enhanced": True,
                "query_type": enhancement_result.get("query_type"),
                "variations": enhancement_result.get("variations", []),
                "sub_queries": enhancement_result.get("sub_queries", []),
                "hyde": enhancement_result.get("hyde", ""),
                "expanded": enhancement_result.get("expanded", query),
            })

            if eval_context:
                eval_context["used_query_enhancement"] = True
                eval_context["used_hyde"] = bool(enhancement_result.get("hyde"))

            logger.info(f"Query enhanced: type={enhancement_result.get('query_type')}")

        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")

        return enhanced_data

    def _perform_retrieval(
        self,
        enhanced_query_data: Dict[str, Any],
        eval_context: Optional[Dict]
    ) -> List[Tuple[str, str, Dict, float]]:
        """Perform retrieval using the best available method"""

        original_query = enhanced_query_data["original_query"]

        # RAG-Fusion (highest priority if enabled)
        if self.config.USE_RAG_FUSION and self.rag_fusion:
            try:
                results = self.rag_fusion.adaptive_fusion_search(
                    original_query,
                    num_variations=self.config.RAG_FUSION_QUERIES,
                    fusion_method=self.config.RAG_FUSION_METHOD,
                    final_top_k=self.config.RERANK_TOP_K
                )
                enhanced_query_data["retrieval_method"] = "rag_fusion"
                if eval_context:
                    eval_context["fusion_method"] = self.config.RAG_FUSION_METHOD
                return results
            except Exception as e:
                logger.error(f"RAG-Fusion failed, falling back: {e}")

        # Hybrid Search (second priority)
        if self.config.USE_HYBRID_SEARCH:
            try:
                results = self.vector_store.hybrid_search(
                    original_query,
                    semantic_weight=self.config.SEMANTIC_WEIGHT,
                    keyword_weight=self.config.KEYWORD_WEIGHT,
                    max_results=self.config.RERANK_TOP_K
                )
                enhanced_query_data["retrieval_method"] = "hybrid"
                return results
            except Exception as e:
                logger.error(f"Hybrid search failed, falling back: {e}")

        # Semantic Search (fallback)
        try:
            results = self.vector_store.search_course_content(
                original_query,
                max_results=self.config.RERANK_TOP_K
            )
            enhanced_query_data["retrieval_method"] = "semantic"
            return results
        except Exception as e:
            logger.error(f"All retrieval methods failed: {e}")
            return []

    def _apply_reranking(
        self,
        query: str,
        retrieval_results: List[Tuple[str, str, Dict, float]],
        eval_context: Optional[Dict]
    ) -> List[Tuple[str, str, Dict, float]]:
        """Apply reranking if enabled"""

        if not self.config.USE_RERANKING or not self.reranker_manager or not retrieval_results:
            return retrieval_results[:self.config.MAX_RESULTS]

        try:
            # Extract documents and scores for reranking
            documents = [content for _, content, _, _ in retrieval_results]
            original_scores = [score for _, _, _, score in retrieval_results]

            # Apply reranking
            reranked_indices_scores = self.reranker_manager.rerank(
                query,
                documents,
                original_scores,
                strategy=self.config.RERANK_STRATEGY,
                top_k=self.config.MAX_RESULTS
            )

            # Reconstruct results with new ordering
            reranked_results = []
            for idx, new_score in reranked_indices_scores:
                if idx < len(retrieval_results):
                    doc_id, content, metadata, _ = retrieval_results[idx]
                    reranked_results.append((doc_id, content, metadata, new_score))

            if eval_context:
                eval_context["used_reranking"] = True

            logger.info(f"Reranked {len(reranked_results)} results using {self.config.RERANK_STRATEGY}")
            return reranked_results

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return retrieval_results[:self.config.MAX_RESULTS]

    def _generate_answer(
        self,
        query: str,
        retrieval_results: List[Tuple[str, str, Dict, float]],
        history: Optional[List]
    ) -> Tuple[str, List[str], List[str]]:
        """Generate answer from retrieval results"""

        if not retrieval_results:
            return "I couldn't find relevant information to answer your question.", [], []

        # Format context from retrieval results
        context_parts = []
        sources = []
        source_links = []

        for doc_id, content, metadata, score in retrieval_results:
            context_parts.append(f"Source: {content}")
            sources.append(content[:200] + "..." if len(content) > 200 else content)

            # Get source links
            course_title = metadata.get("course_title", "")
            lesson_number = metadata.get("lesson_number")

            if course_title:
                if lesson_number is not None:
                    link = self.vector_store.get_lesson_link(course_title, lesson_number)
                else:
                    link = self.vector_store.get_course_link(course_title)
                source_links.append(link)
            else:
                source_links.append(None)

        # Create enhanced prompt with context
        context = "\n\n".join(context_parts)
        prompt = f"""Based on the following course materials, answer this question: {query}

Course Materials:
{context}

Please provide a comprehensive answer based on the provided materials. If you reference specific information, indicate which source it comes from."""

        # Generate response
        try:
            response = self.ai_generator.generate(prompt, max_tokens=1000)
            return response, sources, source_links
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"I found relevant information but couldn't generate a proper response: {e}", sources, source_links

    def _legacy_query_pipeline(
        self,
        query: str,
        history: Optional[List],
        eval_context: Optional[Dict]
    ) -> Tuple[str, List[str], List[str]]:
        """Legacy tool-based query processing for compatibility"""

        # Create prompt for the AI with clear instructions
        prompt = f"""Answer this question about course materials: {query}"""

        # Generate response using AI with tools
        response = self.ai_generator.generate_response(
            query=prompt,
            conversation_history=history,
            tools=self.tool_manager.get_tool_definitions(),
            tool_manager=self.tool_manager,
        )

        # Get sources and source links from the search tool
        sources = self.tool_manager.get_last_sources()
        source_links = self.tool_manager.get_last_source_links()

        # Reset sources after retrieving them
        self.tool_manager.reset_sources()

        if eval_context:
            eval_context["retrieval_method"] = "tool_based"
            eval_context["num_results"] = len(sources)
            eval_context["sources_count"] = len(sources)

        return response, sources, source_links

    def get_course_analytics(self) -> Dict:
        """Get comprehensive analytics about the course catalog and system performance"""
        analytics = {
            "total_courses": self.vector_store.get_course_count(),
            "course_titles": self.vector_store.get_existing_course_titles(),
        }

        # Add system configuration info
        analytics["system_config"] = {
            "hybrid_search_enabled": self.config.USE_HYBRID_SEARCH,
            "query_enhancement_enabled": self.config.USE_QUERY_ENHANCEMENT,
            "reranking_enabled": self.config.USE_RERANKING,
            "rag_fusion_enabled": self.config.USE_RAG_FUSION,
            "evaluation_enabled": self.config.ENABLE_METRICS,
        }

        # Add performance metrics if available
        if self.evaluator:
            try:
                performance_summary = self.evaluator.get_performance_summary()
                if "error" not in performance_summary:
                    analytics["performance_metrics"] = performance_summary
            except Exception as e:
                logger.error(f"Error getting performance metrics: {e}")

        return analytics

    def get_system_status(self) -> Dict[str, Any]:
        """Get detailed system status and health information"""
        status = {
            "system_type": "Enhanced RAG System",
            "components": {
                "vector_store": "active",
                "ai_generator": "active",
                "session_manager": "active",
                "document_processor": "active",
            },
            "advanced_features": {},
            "configuration": {},
        }

        # Check advanced component status
        status["advanced_features"] = {
            "hybrid_search": "active" if self.hybrid_search else "disabled",
            "query_enhancement": "active" if self.query_enhancer else "disabled",
            "reranking": "active" if self.reranker_manager else "disabled",
            "rag_fusion": "active" if self.rag_fusion else "disabled",
            "evaluation": "active" if self.evaluator else "disabled",
        }

        # Add key configuration parameters
        status["configuration"] = {
            "chunk_size": self.config.CHUNK_SIZE,
            "max_results": self.config.MAX_RESULTS,
            "semantic_weight": self.config.SEMANTIC_WEIGHT,
            "keyword_weight": self.config.KEYWORD_WEIGHT,
            "rerank_strategy": self.config.RERANK_STRATEGY,
        }

        # Add data statistics
        try:
            course_count = self.vector_store.get_course_count()
            status["data_statistics"] = {
                "total_courses": course_count,
                "bm25_index_ready": hasattr(self.vector_store, 'bm25_index') and self.vector_store.bm25_index is not None,
            }
        except Exception as e:
            status["data_statistics"] = {"error": str(e)}

        return status

    def export_system_analytics(self, output_file: str = "system_analytics.json") -> Dict:
        """Export comprehensive system analytics to file"""
        analytics = {
            "system_status": self.get_system_status(),
            "course_analytics": self.get_course_analytics(),
            "timestamp": "2025-01-13T00:00:00",  # Will be updated by datetime if needed
        }

        # Add performance analysis if evaluator is available
        if self.evaluator:
            try:
                analytics["performance_analysis"] = self.evaluator.analyze_query_patterns()
                analytics["recent_queries"] = [q.to_dict() for q in self.evaluator.get_recent_queries(10)]
            except Exception as e:
                logger.error(f"Error getting performance analysis: {e}")

        # Export to file
        try:
            import json
            from datetime import datetime
            analytics["timestamp"] = datetime.now().isoformat()

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(analytics, f, indent=2, ensure_ascii=False)

            logger.info(f"System analytics exported to {output_file}")
        except Exception as e:
            logger.error(f"Error exporting analytics: {e}")

        return analytics

    def toggle_advanced_features(self, **feature_flags) -> Dict[str, Any]:
        """Dynamically toggle advanced features (for testing/debugging)"""
        results = {"changes": [], "current_status": {}}

        for feature, enabled in feature_flags.items():
            if feature == "hybrid_search" and hasattr(self.config, "USE_HYBRID_SEARCH"):
                self.config.USE_HYBRID_SEARCH = enabled
                results["changes"].append(f"Hybrid search: {enabled}")

            elif feature == "query_enhancement" and hasattr(self.config, "USE_QUERY_ENHANCEMENT"):
                self.config.USE_QUERY_ENHANCEMENT = enabled
                results["changes"].append(f"Query enhancement: {enabled}")

            elif feature == "reranking" and hasattr(self.config, "USE_RERANKING"):
                self.config.USE_RERANKING = enabled
                results["changes"].append(f"Reranking: {enabled}")

            elif feature == "rag_fusion" and hasattr(self.config, "USE_RAG_FUSION"):
                self.config.USE_RAG_FUSION = enabled
                results["changes"].append(f"RAG-Fusion: {enabled}")

        # Return current status
        results["current_status"] = self.get_system_status()["advanced_features"]

        return results


# Backward compatibility alias
RAGSystem = EnhancedRAGSystem
