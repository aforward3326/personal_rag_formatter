import os
import json
import pytest
import asyncio
import logging
from unittest.mock import patch, AsyncMock
from tenacity import RetryError
import concurrent.futures
from dotenv import load_dotenv

# 1. Load mock.env to override default environment variables before loading the target module
# Ensure the mock.env from the parent directory is read
mock_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mock.env'))
load_dotenv(mock_env_path, override=True)

# Remove the system's GOOGLE_API_KEY to prevent the Google SDK from seeing two keys and printing warnings
if "GOOGLE_API_KEY" in os.environ:
    del os.environ["GOOGLE_API_KEY"]

# Dynamically add the rag_pipeline_model directory to Python's module search path
import sys
target_module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rag_pipeline_model'))
if target_module_dir not in sys.path:
    sys.path.insert(0, target_module_dir)

# 2. Import the target module for testing
# Use patch to suspend load_dotenv to avoid loading the real GOOGLE_API_KEY in the main program
with patch('dotenv.load_dotenv'):
    import rag_ingestion_pipeline

@pytest.fixture
def test_chunk_file():
    """Creates a temporary .pending JSON file to mock chunked data and saves it in the mock_data directory."""
    # Use the CACHE_DIR defined in mock.env
    cache_dir = os.environ.get("CACHE_DIR", "../test/mock_data/cache")
    process_cache_dir = os.path.join(cache_dir, rag_ingestion_pipeline.PROCESS_NAME)
    os.makedirs(process_cache_dir, exist_ok=True)
    
    file_path = os.path.join(process_cache_dir, "rag_ingestion_pipeline_2026071402_chunk_0.json.pending")
    file_content = {
        "group_id": "test_group",
        "content_hash": "test_hash",
        "main_content": "This is a test content intended for the LLM."
    }
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(file_content, f, ensure_ascii=False)
        
    yield file_path
    
    # Post-test cleanup (removes .pending, .processing, .error residual files to keep the test environment clean)
    for ext in [".pending", ".processing", ".error"]:
        cleanup_path = file_path.replace(".pending", ext)
        if os.path.exists(cleanup_path):
            os.remove(cleanup_path)

@pytest.mark.asyncio
async def test_process_single_cache_retry_type_error(test_chunk_file, caplog):
    """
    Test Target: Reproduce RetryError[<Future at ... raised TypeError>] in the log 
    and verify process_single_cache can catch it correctly and mark the file as .error
    """
    semaphore = asyncio.Semaphore(2)

    # 1. Create a Future internally containing a TypeError. 
    # This simulates Tenacity giving up after multiple retries.
    future = concurrent.futures.Future()
    future.set_exception(TypeError("unexpected keyword argument 'generation_config'"))
    
    # 2. Mock the LLM analysis function to force it to raise our prepared RetryError
    with patch.object(rag_ingestion_pipeline, 'analyze_with_llm_for_cleaning', new_callable=AsyncMock) as mock_analyze:
        mock_analyze.side_effect = RetryError(last_attempt=future)
        
        # Capture error-level logs for subsequent verification
        with caplog.at_level(logging.ERROR):
            # Execute the function under test
            await rag_ingestion_pipeline.process_single_cache(test_chunk_file, semaphore)
            
    # 3. Assertion 1: Verify if the file state transitions match expectations (.pending -> .processing -> .error)
    assert not os.path.exists(test_chunk_file), "The original .pending file should be renamed"
    assert not os.path.exists(test_chunk_file.replace(".pending", ".processing")), "The file should not remain in .processing state upon an error"
    assert os.path.exists(test_chunk_file.replace(".pending", ".error")), "The file should be correctly renamed to .error after an error"
    
    # 4. Assertion 2: Check if the log caught and recorded RetryError and TypeError
    assert "Error processing" in caplog.text
    assert "Max retries reached" in caplog.text
    assert "TypeError" in caplog.text

@pytest.mark.asyncio
async def test_real_gemini_api_call():
    """
    Test Target: Real Gemini API call to verify API status and Pydantic structured output. 
    Note: This requires a 'real' GEMINI_API_KEY in mock.env to succeed.
    """
    # Read external test data file (gemini_test_data.json)
    mock_data_path = os.path.join(os.path.dirname(__file__), "mock_data", "gemini_test_data.json")
    with open(mock_data_path, "r", encoding="utf-8") as f:
        chunk_data = json.load(f)

    semaphore = asyncio.Semaphore(1)
    
    # Do not use mock here; call the real LLM processing function directly
    result = await rag_ingestion_pipeline.analyze_with_llm_for_cleaning(chunk_data, semaphore)
    
    # Assertion: Ensure the returned result is successfully parsed into our defined Pydantic model
    assert isinstance(result, rag_ingestion_pipeline.CleanedContentResult), "The returned result is not a valid CleanedContentResult"
    assert hasattr(result.analysis, "is_noise"), "The parsed JSON is missing the is_noise attribute"
    
    # Print the result to display the real LLM response
    print(f"\n[Gemini API Response]:\n{result}")

if __name__ == "__main__":
    pytest.main([
        "-v",
        "-s",
        __file__
    ])