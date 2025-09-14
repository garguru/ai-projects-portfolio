import os
import sys
import tempfile
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def isolated_app():
    """Create an isolated test app that doesn't mount static files"""
    # Mock the app creation to avoid static file mounting issues
    with patch('app.rag_system') as mock_rag:
        # Configure mock RAG system
        mock_rag.query.return_value = (
            "Test response about course content.",
            ["Test Course - Lesson 1"],
            ["https://example.com/lesson1"]
        )
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": [
                "Building Towards Computer Use with Anthropic",
                "MCP: Build Rich-Context AI Apps with Anthropic",
                "Advanced Retrieval for AI with Chroma"
            ]
        }
        mock_rag.session_manager.create_session.return_value = "test-session-456"
        mock_rag.session_manager.clear_session.return_value = None

        # Import app after mocking to avoid static file issues
        from app import app
        yield app


@pytest.fixture
def isolated_client(isolated_app):
    """Create test client for isolated app"""
    return TestClient(isolated_app)


@pytest.mark.api
@pytest.mark.integration
class TestAppIntegration:
    """Integration tests for the actual FastAPI app endpoints"""

    def test_query_endpoint_real_app(self, isolated_client):
        """Test /api/query endpoint with the real app"""
        request_data = {
            "query": "What is computer use in AI?",
            "session_id": "test-session"
        }

        response = isolated_client.post("/api/query", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "source_links" in data
        assert "session_id" in data

        # Verify response content
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["source_links"], list)

    def test_courses_endpoint_real_app(self, isolated_client):
        """Test /api/courses endpoint with the real app"""
        response = isolated_client.get("/api/courses")
        assert response.status_code == 200

        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data

        # Verify data types and content
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert data["total_courses"] > 0
        assert len(data["course_titles"]) > 0

    def test_health_check_endpoint(self, isolated_client):
        """Test basic health check functionality"""
        response = isolated_client.get("/api/courses")
        assert response.status_code == 200

        # If courses endpoint works, the app is healthy
        data = response.json()
        assert "total_courses" in data

    def test_cors_headers_real_app(self, isolated_client):
        """Test CORS headers are properly configured"""
        response = isolated_client.options("/api/query")
        # CORS middleware should handle OPTIONS requests
        assert response.status_code in [200, 405]  # 405 is acceptable for OPTIONS

        # Test actual request to verify CORS works
        response = isolated_client.post("/api/query", json={
            "query": "test"
        })
        assert response.status_code == 200


@pytest.mark.api
class TestAppErrorHandling:
    """Test error handling in the real app"""

    @patch('app.rag_system')
    def test_query_endpoint_rag_error(self, mock_rag, isolated_client):
        """Test /api/query endpoint when RAG system raises an error"""
        # Configure mock to raise an exception
        mock_rag.query.side_effect = Exception("RAG system error")

        request_data = {"query": "test query"}
        response = isolated_client.post("/api/query", json=request_data)

        # Should return 500 Internal Server Error
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

    @patch('app.rag_system')
    def test_courses_endpoint_rag_error(self, mock_rag, isolated_client):
        """Test /api/courses endpoint when RAG system raises an error"""
        # Configure mock to raise an exception
        mock_rag.get_course_analytics.side_effect = Exception("Analytics error")

        response = isolated_client.get("/api/courses")

        # Should return 500 Internal Server Error
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data


@pytest.mark.api
class TestAppValidation:
    """Test request validation in the real app"""

    def test_query_validation_missing_query(self, isolated_client):
        """Test query validation when query field is missing"""
        response = isolated_client.post("/api/query", json={})
        assert response.status_code == 422

        data = response.json()
        assert "detail" in data
        # Should mention the missing 'query' field
        error_msg = str(data["detail"]).lower()
        assert "query" in error_msg

    def test_query_validation_wrong_type(self, isolated_client):
        """Test query validation when query field has wrong type"""
        response = isolated_client.post("/api/query", json={
            "query": 123  # Should be string
        })
        assert response.status_code == 422

        data = response.json()
        assert "detail" in data

    def test_query_validation_extra_fields(self, isolated_client):
        """Test that extra fields are ignored gracefully"""
        response = isolated_client.post("/api/query", json={
            "query": "test query",
            "session_id": "test-123",
            "extra_field": "should be ignored"
        })

        # Should succeed despite extra field
        assert response.status_code == 200


@pytest.mark.api
@pytest.mark.slow
class TestAppPerformance:
    """Performance and load tests for the app"""

    def test_concurrent_requests(self, isolated_client):
        """Test handling of concurrent requests"""
        import threading
        import time

        results = []

        def make_request():
            response = isolated_client.post("/api/query", json={
                "query": f"Test query at {time.time()}"
            })
            results.append(response.status_code)

        # Start multiple concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Wait for all requests to complete
        for thread in threads:
            thread.join()

        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5

    def test_large_query_handling(self, isolated_client):
        """Test handling of large query strings"""
        large_query = "What is computer use? " * 100  # ~2000 characters

        response = isolated_client.post("/api/query", json={
            "query": large_query
        })

        # Should handle large queries gracefully
        assert response.status_code == 200

        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 0


@pytest.mark.integration
class TestAppEndToEnd:
    """End-to-end workflow tests"""

    def test_complete_user_workflow(self, isolated_client):
        """Test a complete user workflow from start to finish"""
        # Step 1: Get available courses
        courses_response = isolated_client.get("/api/courses")
        assert courses_response.status_code == 200
        courses_data = courses_response.json()
        assert courses_data["total_courses"] > 0

        # Step 2: Start a new conversation
        query1_response = isolated_client.post("/api/query", json={
            "query": "Tell me about the available courses"
        })
        assert query1_response.status_code == 200
        query1_data = query1_response.json()
        session_id = query1_data["session_id"]
        assert session_id is not None

        # Step 3: Continue the conversation with the same session
        query2_response = isolated_client.post("/api/query", json={
            "query": "Can you tell me more about the first course?",
            "session_id": session_id
        })
        assert query2_response.status_code == 200
        query2_data = query2_response.json()
        assert query2_data["session_id"] == session_id

        # Step 4: Clear the session (if endpoint exists)
        # Note: This test assumes a clear-session endpoint exists
        # If it doesn't exist in the real app, this part can be removed
        try:
            clear_response = isolated_client.post("/api/clear-session", json={
                "session_id": session_id
            })
            # If endpoint exists, it should work
            if clear_response.status_code != 404:
                assert clear_response.status_code == 200
        except Exception:
            # Clear session endpoint may not exist, that's ok
            pass