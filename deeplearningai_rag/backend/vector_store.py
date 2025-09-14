from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import pickle
import os

import chromadb
from chromadb.config import Settings
from models import Course, CourseChunk
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


@dataclass
class SearchResults:
    """Container for search results with metadata"""

    documents: List[str]
    metadata: List[Dict[str, Any]]
    distances: List[float]
    error: Optional[str] = None

    @classmethod
    def from_chroma(cls, chroma_results: Dict) -> "SearchResults":
        """Create SearchResults from ChromaDB query results"""
        return cls(
            documents=(
                chroma_results["documents"][0] if chroma_results["documents"] else []
            ),
            metadata=(
                chroma_results["metadatas"][0] if chroma_results["metadatas"] else []
            ),
            distances=(
                chroma_results["distances"][0] if chroma_results["distances"] else []
            ),
        )

    @classmethod
    def empty(cls, error_msg: str) -> "SearchResults":
        """Create empty results with error message"""
        return cls(documents=[], metadata=[], distances=[], error=error_msg)

    def is_empty(self) -> bool:
        """Check if results are empty"""
        return len(self.documents) == 0


class VectorStore:
    """Vector storage using ChromaDB for course content and metadata with BM25 support"""

    def __init__(self, chroma_path: str, embedding_model: str, max_results: int = 5):
        self.max_results = max_results
        self.chroma_path = chroma_path

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=chroma_path, settings=Settings(anonymized_telemetry=False)
        )

        # Set up sentence transformer embedding function
        self.embedding_function = (
            chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        )

        # Create collections for different types of data
        self.course_catalog = self._create_collection(
            "course_catalog"
        )  # Course titles/instructors
        self.course_content = self._create_collection(
            "course_content"
        )  # Actual course material

        # BM25 support
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_metadata = []
        self.bm25_index_path = os.path.join(chroma_path, "bm25_index.pkl")
        self.bm25_docs_path = os.path.join(chroma_path, "bm25_docs.pkl")

        # Load existing BM25 index if available
        self._load_bm25_index()

    def _create_collection(self, name: str):
        """Create or get a ChromaDB collection"""
        return self.client.get_or_create_collection(
            name=name, embedding_function=self.embedding_function
        )

    def search(
        self,
        query: str,
        course_name: Optional[str] = None,
        lesson_number: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> SearchResults:
        """
        Main search interface that handles course resolution and content search.

        Args:
            query: What to search for in course content
            course_name: Optional course name/title to filter by
            lesson_number: Optional lesson number to filter by
            limit: Maximum results to return

        Returns:
            SearchResults object with documents and metadata
        """
        # Step 1: Resolve course name if provided
        course_title = None
        if course_name:
            course_title = self._resolve_course_name(course_name)
            if not course_title:
                return SearchResults.empty(f"No course found matching '{course_name}'")

        # Step 2: Build filter for content search
        filter_dict = self._build_filter(course_title, lesson_number)

        # Step 3: Search course content
        # Use provided limit or fall back to configured max_results
        search_limit = limit if limit is not None else self.max_results

        try:
            results = self.course_content.query(
                query_texts=[query], n_results=search_limit, where=filter_dict
            )
            return SearchResults.from_chroma(results)
        except Exception as e:
            return SearchResults.empty(f"Search error: {str(e)}")

    def _resolve_course_name(self, course_name: str) -> Optional[str]:
        """Use vector search to find best matching course by name"""
        try:
            results = self.course_catalog.query(query_texts=[course_name], n_results=1)

            if results["documents"][0] and results["metadatas"][0]:
                # Return the title (which is now the ID)
                return results["metadatas"][0][0]["title"]
        except Exception as e:
            print(f"Error resolving course name: {e}")

        return None

    def _build_filter(
        self, course_title: Optional[str], lesson_number: Optional[int]
    ) -> Optional[Dict]:
        """Build ChromaDB filter from search parameters"""
        if not course_title and lesson_number is None:
            return None

        # Handle different filter combinations
        if course_title and lesson_number is not None:
            return {
                "$and": [
                    {"course_title": course_title},
                    {"lesson_number": lesson_number},
                ]
            }

        if course_title:
            return {"course_title": course_title}

        return {"lesson_number": lesson_number}

    def add_course_metadata(self, course: Course):
        """Add course information to the catalog for semantic search"""
        import json

        course_text = course.title

        # Build lessons metadata and serialize as JSON string
        lessons_metadata = []
        for lesson in course.lessons:
            lessons_metadata.append(
                {
                    "lesson_number": lesson.lesson_number,
                    "lesson_title": lesson.title,
                    "lesson_link": lesson.lesson_link,
                }
            )

        self.course_catalog.add(
            documents=[course_text],
            metadatas=[
                {
                    "title": course.title,
                    "instructor": course.instructor,
                    "course_link": course.course_link,
                    "lessons_json": json.dumps(
                        lessons_metadata
                    ),  # Serialize as JSON string
                    "lesson_count": len(course.lessons),
                }
            ],
            ids=[course.title],
        )

    def add_course_content(self, chunks: List[CourseChunk]):
        """Add course content chunks to the vector store"""
        if not chunks:
            return

        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "course_title": chunk.course_title,
                "lesson_number": chunk.lesson_number,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]
        # Use title with chunk index for unique IDs
        ids = [
            f"{chunk.course_title.replace(' ', '_')}_{chunk.chunk_index}"
            for chunk in chunks
        ]

        self.course_content.add(documents=documents, metadatas=metadatas, ids=ids)

    def clear_all_data(self):
        """Clear all data from both collections"""
        try:
            self.client.delete_collection("course_catalog")
            self.client.delete_collection("course_content")
            # Recreate collections
            self.course_catalog = self._create_collection("course_catalog")
            self.course_content = self._create_collection("course_content")
        except Exception as e:
            print(f"Error clearing data: {e}")

    def get_existing_course_titles(self) -> List[str]:
        """Get all existing course titles from the vector store"""
        try:
            # Get all documents from the catalog
            results = self.course_catalog.get()
            if results and "ids" in results:
                return results["ids"]
            return []
        except Exception as e:
            print(f"Error getting existing course titles: {e}")
            return []

    def get_course_count(self) -> int:
        """Get the total number of courses in the vector store"""
        try:
            results = self.course_catalog.get()
            if results and "ids" in results:
                return len(results["ids"])
            return 0
        except Exception as e:
            print(f"Error getting course count: {e}")
            return 0

    def get_all_courses_metadata(self) -> List[Dict[str, Any]]:
        """Get metadata for all courses in the vector store"""
        import json

        try:
            results = self.course_catalog.get()
            if results and "metadatas" in results:
                # Parse lessons JSON for each course
                parsed_metadata = []
                for metadata in results["metadatas"]:
                    course_meta = metadata.copy()
                    if "lessons_json" in course_meta:
                        course_meta["lessons"] = json.loads(course_meta["lessons_json"])
                        del course_meta[
                            "lessons_json"
                        ]  # Remove the JSON string version
                    parsed_metadata.append(course_meta)
                return parsed_metadata
            return []
        except Exception as e:
            print(f"Error getting courses metadata: {e}")
            return []

    def get_course_link(self, course_title: str) -> Optional[str]:
        """Get course link for a given course title"""
        try:
            # Get course by ID (title is the ID)
            results = self.course_catalog.get(ids=[course_title])
            if results and "metadatas" in results and results["metadatas"]:
                metadata = results["metadatas"][0]
                return metadata.get("course_link")
            return None
        except Exception as e:
            print(f"Error getting course link: {e}")
            return None

    def get_lesson_link(self, course_title: str, lesson_number: int) -> Optional[str]:
        """Get lesson link for a given course title and lesson number"""
        import json

        try:
            # Get course by ID (title is the ID)
            results = self.course_catalog.get(ids=[course_title])
            if results and "metadatas" in results and results["metadatas"]:
                metadata = results["metadatas"][0]
                lessons_json = metadata.get("lessons_json")
                if lessons_json:
                    lessons = json.loads(lessons_json)
                    # Find the lesson with matching number
                    for lesson in lessons:
                        if lesson.get("lesson_number") == lesson_number:
                            return lesson.get("lesson_link")
            return None
        except Exception as e:
            print(f"Error getting lesson link: {e}")
            return None

    # BM25 Methods

    def _load_bm25_index(self):
        """Load existing BM25 index from disk if available"""
        try:
            if os.path.exists(self.bm25_index_path) and os.path.exists(self.bm25_docs_path):
                with open(self.bm25_index_path, "rb") as f:
                    self.bm25_index = pickle.load(f)
                with open(self.bm25_docs_path, "rb") as f:
                    data = pickle.load(f)
                    self.bm25_documents = data.get("documents", [])
                    self.bm25_metadata = data.get("metadata", [])
                print(f"Loaded BM25 index with {len(self.bm25_documents)} documents")
        except Exception as e:
            print(f"Error loading BM25 index: {e}")
            self.bm25_index = None
            self.bm25_documents = []
            self.bm25_metadata = []

    def _save_bm25_index(self):
        """Save BM25 index to disk"""
        try:
            os.makedirs(os.path.dirname(self.bm25_index_path), exist_ok=True)

            if self.bm25_index:
                with open(self.bm25_index_path, "wb") as f:
                    pickle.dump(self.bm25_index, f)

                with open(self.bm25_docs_path, "wb") as f:
                    pickle.dump({
                        "documents": self.bm25_documents,
                        "metadata": self.bm25_metadata
                    }, f)
                print(f"Saved BM25 index with {len(self.bm25_documents)} documents")
        except Exception as e:
            print(f"Error saving BM25 index: {e}")

    def _rebuild_bm25_index(self):
        """Rebuild BM25 index from current ChromaDB content"""
        try:
            # Get all documents from ChromaDB
            results = self.course_content.get()
            if not results or not results.get("documents"):
                print("No documents found to build BM25 index")
                return

            documents = results["documents"]
            metadatas = results.get("metadatas", [])

            # Tokenize documents for BM25
            tokenized_docs = [doc.lower().split() for doc in documents]

            # Build BM25 index
            self.bm25_index = BM25Okapi(tokenized_docs)
            self.bm25_documents = documents
            self.bm25_metadata = metadatas

            # Save to disk
            self._save_bm25_index()
            print(f"Built BM25 index with {len(documents)} documents")

        except Exception as e:
            print(f"Error building BM25 index: {e}")

    def build_bm25_index(self):
        """Public method to build/rebuild BM25 index"""
        self._rebuild_bm25_index()

    def bm25_search(
        self,
        query: str,
        top_k: int = 10,
        course_title: Optional[str] = None,
        lesson_number: Optional[int] = None
    ) -> List[Tuple[str, str, Dict, float]]:
        """
        Perform BM25 keyword search.

        Args:
            query: Search query
            top_k: Number of results to return
            course_title: Optional course filter
            lesson_number: Optional lesson filter

        Returns:
            List of (doc_id, content, metadata, score) tuples
        """
        if not self.bm25_index or not self.bm25_documents:
            print("BM25 index not available, building it now...")
            self._rebuild_bm25_index()
            if not self.bm25_index:
                return []

        try:
            # Tokenize query
            tokenized_query = query.lower().split()

            # Get BM25 scores
            scores = self.bm25_index.get_scores(tokenized_query)

            # Get top-k indices
            top_indices = scores.argsort()[-top_k:][::-1]

            results = []
            for idx in top_indices:
                if idx < len(self.bm25_documents):
                    content = self.bm25_documents[idx]
                    metadata = self.bm25_metadata[idx] if idx < len(self.bm25_metadata) else {}
                    score = float(scores[idx])

                    # Apply filters if specified
                    if course_title and metadata.get("course_title") != course_title:
                        continue
                    if lesson_number is not None and metadata.get("lesson_number") != lesson_number:
                        continue

                    # Generate doc_id from metadata
                    doc_id = f"{metadata.get('course_title', 'unknown')}_{metadata.get('chunk_index', idx)}"

                    results.append((doc_id, content, metadata, score))

            return results

        except Exception as e:
            print(f"Error in BM25 search: {e}")
            return []

    def search_course_content(
        self,
        query: str,
        max_results: Optional[int] = None,
        course_name: Optional[str] = None,
        lesson_number: Optional[int] = None
    ) -> List[Tuple[str, str, Dict, float]]:
        """
        Search course content using semantic search.
        Returns results in format compatible with hybrid search.

        Args:
            query: Search query
            max_results: Maximum results to return
            course_name: Optional course filter
            lesson_number: Optional lesson filter

        Returns:
            List of (doc_id, content, metadata, score) tuples
        """
        search_results = self.search(query, course_name, lesson_number, max_results)

        results = []
        for i, (content, metadata, distance) in enumerate(zip(
            search_results.documents,
            search_results.metadata,
            search_results.distances
        )):
            # Convert distance to similarity score (1 - distance)
            score = max(0.0, 1.0 - distance)

            # Generate doc_id
            doc_id = f"{metadata.get('course_title', 'unknown')}_{metadata.get('chunk_index', i)}"

            results.append((doc_id, content, metadata, score))

        return results

    def hybrid_search(
        self,
        query: str,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        max_results: int = 10,
        course_name: Optional[str] = None,
        lesson_number: Optional[int] = None
    ) -> List[Tuple[str, str, Dict, float]]:
        """
        Perform hybrid search combining semantic and BM25 results.

        Args:
            query: Search query
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for BM25 search (0-1)
            max_results: Maximum results to return
            course_name: Optional course filter
            lesson_number: Optional lesson filter

        Returns:
            List of (doc_id, content, metadata, combined_score) tuples
        """
        # Normalize weights
        total_weight = semantic_weight + keyword_weight
        sem_w = semantic_weight / total_weight
        key_w = keyword_weight / total_weight

        # Get semantic search results
        semantic_results = self.search_course_content(
            query, max_results * 2, course_name, lesson_number
        )

        # Get BM25 search results
        bm25_results = self.bm25_search(
            query, max_results * 2, course_name, lesson_number
        )

        # Combine results
        combined_scores = {}
        doc_contents = {}

        # Process semantic results
        if semantic_results:
            sem_scores = [score for _, _, _, score in semantic_results]
            max_sem = max(sem_scores) if sem_scores else 1.0

            for doc_id, content, metadata, score in semantic_results:
                normalized_score = score / max_sem if max_sem > 0 else 0
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + (normalized_score * sem_w)
                doc_contents[doc_id] = (content, metadata)

        # Process BM25 results
        if bm25_results:
            bm25_scores = [score for _, _, _, score in bm25_results]
            max_bm25 = max(bm25_scores) if bm25_scores else 1.0

            for doc_id, content, metadata, score in bm25_results:
                normalized_score = score / max_bm25 if max_bm25 > 0 else 0
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + (normalized_score * key_w)
                doc_contents[doc_id] = (content, metadata)

        # Sort and return top results
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        for doc_id, score in sorted_results[:max_results]:
            if doc_id in doc_contents:
                content, metadata = doc_contents[doc_id]
                final_results.append((doc_id, content, metadata, score))

        return final_results
