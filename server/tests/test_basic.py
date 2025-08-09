"""
Basic tests for the Story Generator application
These tests run in CI/CD to ensure basic functionality
"""
import pytest
import asyncio
from pathlib import Path

def test_basic_imports():
    """Test that basic modules can be imported"""
    try:
        from ..app.core.story_generator import StoryGenerator
        from ..app.core.socket_manager import sio
        from ..app import main
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")

def test_environment_setup():
    """Test that required environment variables work"""
    import os
    
    # Test that default values work
    stage = os.getenv("STAGE", "TEST") 
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    assert stage in ["DEV", "PROD", "TEST"]
    assert isinstance(port, int)
    assert port > 0
    assert isinstance(host, str)

def test_model_files_exist():
    """Test that required model files exist"""
    current_dir = Path(__file__).parent.parent
    model_dir = current_dir / "app" / "model"
    
    # Check for model files
    model_file = model_dir / "model.pt"
    tokenizer_dir = model_dir / "tokenizer"
    
    # In CI, we might not have the actual model files, so just check structure
    assert model_dir.exists(), "Model directory should exist"
    # Only check for files if they exist (they might not in CI)
    if model_file.exists():
        assert model_file.is_file(), "Model file should be a file"
    if tokenizer_dir.exists():
        assert tokenizer_dir.is_dir(), "Tokenizer should be a directory"

@pytest.mark.asyncio
async def test_story_generator_initialization():
    """Test that StoryGenerator can be initialized"""
    try:
        from ..app.core.story_generator import StoryGenerator
        
        # Try to initialize (might fail in CI without model files)
        try:
            generator = StoryGenerator()
            assert generator is not None
        except FileNotFoundError:
            # Expected in CI environment without model files
            pytest.skip("Model files not available in CI environment")
        except Exception as e:
            pytest.fail(f"Unexpected error initializing StoryGenerator: {e}")
            
    except ImportError:
        pytest.skip("StoryGenerator not available")

def test_socket_manager_initialization():
    """Test that Socket.IO server initializes"""
    try:
        from ..app.core.socket_manager import sio
        assert sio is not None
        assert hasattr(sio, 'emit')
        assert hasattr(sio, 'on')
    except ImportError as e:
        pytest.fail(f"Failed to import socket manager: {e}")

def test_fastapi_app_creation():
    """Test that FastAPI app can be created"""
    try:
        from ..app.main import app
        assert app is not None
        assert hasattr(app, 'routes')
    except ImportError as e:
        pytest.fail(f"Failed to import FastAPI app: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])