#### What this tests ####
#    This tests the router's handling of None metadata in lowest latency routing

import os
import sys
import time
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_latency import LowestLatencyLoggingHandler


def test_none_metadata_no_error():
    """
    Test that log_success_event handles None metadata without AttributeError
    
    This tests the fix for the issue where None metadata (common with aliases)
    causes "'NoneType' object has no attribute 'get'" error
    """
    test_cache = DualCache()

    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache
    )
    
    deployment_id = "1234"
    # Test with None metadata
    kwargs = {
        "litellm_params": {
            "metadata": None,  # This is None, which causes the error
            "model_info": {"id": deployment_id},
        }
    }
    
    response_obj = litellm.ModelResponse(
        id='test-response',
        created=1752669685,
        model='gpt-3.5-turbo',
        object='chat.completion',
        choices=[
            litellm.Choices(
                finish_reason='stop',
                index=0,
                message=litellm.Message(
                    content='Hello',
                    role='assistant',
                    tool_calls=None
                )
            )
        ],
        usage=litellm.Usage(
            completion_tokens=10,
            prompt_tokens=20,
            total_tokens=30
        )
    )
    
    start_time = time.time()
    time.sleep(0.1)
    end_time = time.time()
    
    # This should not raise AttributeError
    try:
        lowest_latency_logger.log_success_event(
            response_obj=response_obj,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )
    except AttributeError as e:
        pytest.fail(f"log_success_event raised AttributeError with None metadata: {str(e)}")
    
    # Verify that nothing was logged (since metadata is None)
    model_group = None
    cached_value = test_cache.get_cache(key=f"{model_group}_map")
    # Should be None or empty since we can't log without metadata
    assert cached_value is None or cached_value == {}


@pytest.mark.asyncio
async def test_async_none_metadata_no_error():
    """
    Test that async_log_success_event handles None metadata without AttributeError
    """
    test_cache = DualCache()
    
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache
    )
    
    deployment_id = "5678"
    # Test with None litellm_metadata
    kwargs = {
        "litellm_params": {
            "litellm_metadata": None,  # This is None, which causes the error
            "model_info": {"id": deployment_id},
        }
    }
    
    response_obj = litellm.ModelResponse(
        usage=litellm.Usage(
            completion_tokens=15,
            prompt_tokens=25,
            total_tokens=40
        )
    )
    
    start_time = time.time()
    time.sleep(0.1)
    end_time = time.time()
    
    # This should not raise AttributeError
    try:
        await lowest_latency_logger.async_log_success_event(
            response_obj=response_obj,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )
    except AttributeError as e:
        pytest.fail(f"async_log_success_event raised AttributeError with None metadata: {str(e)}")


@pytest.mark.asyncio
async def test_async_none_metadata_failure_no_error():
    """
    Test that async_log_failure_event handles None metadata without AttributeError
    """
    test_cache = DualCache()
    
    lowest_latency_logger = LowestLatencyLoggingHandler(
        router_cache=test_cache
    )
    
    deployment_id = "9999"
    # Test with None metadata and Timeout exception
    kwargs = {
        "litellm_params": {
            "metadata": None,  # This is None, which causes the error
            "model_info": {"id": deployment_id},
        },
        "exception": litellm.Timeout(
            message="Request timed out",
            model="gpt-3.5-turbo",
            llm_provider="openai"
        )
    }
    
    response_obj = None
    start_time = time.time()
    time.sleep(0.1)
    end_time = time.time()
    
    # This should not raise AttributeError
    try:
        await lowest_latency_logger.async_log_failure_event(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
        )
    except AttributeError as e:
        pytest.fail(f"async_log_failure_event raised AttributeError with None metadata: {str(e)}")


if __name__ == "__main__":
    test_none_metadata_no_error()
    print("Sync test passed!")
    
    import asyncio
    asyncio.run(test_async_none_metadata_no_error())
    print("Async success test passed!")
    
    asyncio.run(test_async_none_metadata_failure_no_error())
    print("Async failure test passed!")
    
    print("All tests passed!")
