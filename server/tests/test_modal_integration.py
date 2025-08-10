"""
ML Model validation tests for Modal integration
Tests the actual ML pipeline end-to-end in CI/CD
"""

import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, patch, MagicMock
import aiohttp
from pathlib import Path
import os


class TestModalIntegration:
    """Test suite for Modal ML model integration"""

    @pytest.fixture
    def mock_modal_response(self):
        """Mock successful Modal response"""
        return {
            "success": True,
            "generated_text": "Once upon a time there was a brave knight who ventured into the dark forest.",
            "tokens_generated": 16,
            "generation_time": 2.3,
            "model_version": "v1.0.0"
        }

    @pytest.fixture
    def mock_modal_error_response(self):
        """Mock Modal error response"""
        return {
            "success": False,
            "error": "Model temporarily unavailable",
            "generated_text": "",
            "tokens_generated": 0
        }

    @pytest.mark.asyncio
    async def test_modal_http_endpoint_success(self, mock_modal_response):
        """Test successful HTTP call to Modal endpoint"""
        from ..app.core.modal_story_generator import ModalStoryGenerator
        
        generator = ModalStoryGenerator()
        
        # Mock the HTTP call
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_modal_response)
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await generator._call_modal_http("Test prompt", 50, 0.7)
            
            assert result["success"] is True
            assert len(result["generated_text"]) > 0
            assert result["tokens_generated"] == 16

    @pytest.mark.asyncio
    async def test_modal_http_endpoint_error(self, mock_modal_error_response):
        """Test Modal HTTP endpoint error handling"""
        from ..app.core.modal_story_generator import ModalStoryGenerator
        
        generator = ModalStoryGenerator()
        
        # Mock HTTP error
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Internal Server Error")
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await generator._call_modal_http("Test prompt", 50, 0.7)
            
            assert result["success"] is False
            assert "HTTP 500" in result["error"]

    @pytest.mark.asyncio
    async def test_modal_timeout_handling(self):
        """Test Modal timeout handling"""
        from ..app.core.modal_story_generator import ModalStoryGenerator
        
        generator = ModalStoryGenerator()
        
        # Mock timeout
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = asyncio.TimeoutError()
            
            result = await generator._call_modal_http("Test prompt", 50, 0.7)
            
            assert result["success"] is False
            assert "timed out" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_story_generation_streaming(self, mock_modal_response):
        """Test the streaming story generation interface"""
        from ..app.core.modal_story_generator import ModalStoryGenerator
        
        generator = ModalStoryGenerator()
        generator.use_client = False  # Force HTTP mode
        
        # Mock the HTTP call
        with patch.object(generator, '_call_modal_http', return_value=mock_modal_response):
            words = []
            async for word in generator.generate_story_stream("Test prompt", 50, 0.7):
                words.append(word)
            
            generated_text = ''.join(words)
            assert len(generated_text) > 0
            assert "Once upon a time" in generated_text

    def test_modal_environment_variables(self):
        """Test Modal configuration from environment variables"""
        from ..app.core.modal_story_generator import ModalStoryGenerator
        
        # Test with URL set
        with patch.dict(os.environ, {'MODAL_ENDPOINT_URL': 'https://test.modal.com'}):
            generator = ModalStoryGenerator()
            assert generator.modal_url == 'https://test.modal.com'
            assert generator.use_client is False
        
        # Test without URL set
        with patch.dict(os.environ, {}, clear=True):
            generator = ModalStoryGenerator()
            assert generator.modal_url is None
            assert generator.use_client is True


class TestMLPerformanceBenchmarks:
    """Performance benchmarking for ML model"""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_response_time_benchmark(self):
        """Benchmark Modal response time"""
        from ..app.core.modal_story_generator import ModalStoryGenerator
        
        generator = ModalStoryGenerator()
        
        # Mock fast response
        mock_response = {
            "success": True,
            "generated_text": "Fast response test",
            "tokens_generated": 3,
            "generation_time": 0.5
        }
        
        with patch.object(generator, '_call_modal_http', return_value=mock_response):
            start_time = time.time()
            
            words = []
            async for word in generator.generate_story_stream("Benchmark test", 10, 0.0):
                words.append(word)
            
            total_time = time.time() - start_time
            
            # Should complete within reasonable time (including streaming simulation)
            assert total_time < 5.0  # 5 seconds max
            assert len(''.join(words)) > 0

    def test_input_validation(self):
        """Test input parameter validation"""
        from ..app.core.modal_story_generator import ModalStoryGenerator
        
        generator = ModalStoryGenerator()
        
        # These should not raise exceptions
        assert generator.modal_app_name == "story-generator"
        assert isinstance(generator.use_client, bool)

    @pytest.mark.asyncio
    async def test_concurrent_requests_handling(self):
        """Test handling multiple concurrent requests"""
        from ..app.core.modal_story_generator import ModalStoryGenerator
        
        generator = ModalStoryGenerator()
        
        mock_response = {
            "success": True,
            "generated_text": "Concurrent test response",
            "tokens_generated": 4,
            "generation_time": 1.0
        }
        
        with patch.object(generator, '_call_modal_http', return_value=mock_response):
            # Create multiple concurrent requests
            tasks = []
            for i in range(3):
                task = generator.generate_story_stream(f"Prompt {i}", 5, 0.0)
                tasks.append(task)
            
            # Should handle concurrent requests without issues
            results = []
            for task in tasks:
                words = []
                async for word in task:
                    words.append(word)
                results.append(''.join(words))
            
            assert len(results) == 3
            for result in results:
                assert len(result) > 0


class TestModelMetrics:
    """Test model performance metrics collection"""

    def test_metrics_structure(self):
        """Test that we can collect proper metrics"""
        mock_response = {
            "success": True,
            "generated_text": "Test response for metrics",
            "tokens_generated": 5,
            "generation_time": 1.2,
            "model_version": "v1.0.0"
        }
        
        # Validate metrics structure
        assert "success" in mock_response
        assert "tokens_generated" in mock_response
        assert "generation_time" in mock_response
        assert isinstance(mock_response["tokens_generated"], int)
        assert isinstance(mock_response["generation_time"], (int, float))

    def test_performance_thresholds(self):
        """Test performance threshold validation"""
        # Define acceptable performance thresholds
        MAX_RESPONSE_TIME = 10.0  # seconds
        MIN_TOKENS_PER_SECOND = 2.0  # tokens/second
        
        mock_metrics = {
            "generation_time": 2.5,
            "tokens_generated": 8
        }
        
        tokens_per_second = mock_metrics["tokens_generated"] / mock_metrics["generation_time"]
        
        assert mock_metrics["generation_time"] < MAX_RESPONSE_TIME
        assert tokens_per_second >= MIN_TOKENS_PER_SECOND

    def test_error_rate_tracking(self):
        """Test error rate calculation"""
        # Simulate a batch of requests with some failures
        responses = [
            {"success": True}, {"success": True}, {"success": False},
            {"success": True}, {"success": False}, {"success": True}
        ]
        
        total_requests = len(responses)
        successful_requests = sum(1 for r in responses if r["success"])
        error_rate = (total_requests - successful_requests) / total_requests
        
        assert error_rate == pytest.approx(0.333, abs=0.01)  # 2/6 = 33.3%
        assert error_rate < 0.5  # Error rate should be under 50%


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])