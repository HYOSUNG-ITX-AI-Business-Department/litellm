from litellm._redis import (
    REDIS_SOCKET_TIMEOUT,
    _init_async_redis_sentinel,
    _init_redis_sentinel,
    _get_redis_cluster_kwargs,
    get_redis_async_client,
    get_redis_url_from_environment,
)
import pytest
from unittest.mock import MagicMock, patch
import redis.asyncio as async_redis


def test_get_redis_url_from_environment_single_url(monkeypatch):
    """Test when REDIS_URL is directly provided"""
    # Set the environment variable
    monkeypatch.setenv("REDIS_URL", "redis://redis-server:6379/0")

    # Call the function to get the Redis URL
    redis_url = get_redis_url_from_environment()

    # Assert that the returned URL matches the expected value
    assert redis_url == "redis://redis-server:6379/0"


def test_get_redis_url_from_environment_host_port(monkeypatch):
    """Test when REDIS_HOST and REDIS_PORT are provided"""
    # Set the environment variables
    monkeypatch.setenv("REDIS_HOST", "redis-server")
    monkeypatch.setenv("REDIS_PORT", "6379")
    # Ensure authentication variables are not set
    monkeypatch.delenv("REDIS_USERNAME", raising=False)
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)
    monkeypatch.delenv("REDIS_SSL", raising=False)

    # Call the function to get the Redis URL
    redis_url = get_redis_url_from_environment()

    # Assert that the returned URL matches the expected value
    assert redis_url == "redis://redis-server:6379"


def test_get_redis_url_from_environment_with_ssl(monkeypatch):
    """Test when SSL is enabled"""
    # Set the environment variables
    monkeypatch.setenv("REDIS_HOST", "redis-server")
    monkeypatch.setenv("REDIS_PORT", "6379")
    monkeypatch.setenv("REDIS_SSL", "true")
    # Ensure authentication variables are not set
    monkeypatch.delenv("REDIS_USERNAME", raising=False)
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)

    # Call the function to get the Redis URL
    redis_url = get_redis_url_from_environment()

    # Assert that the returned URL uses rediss:// protocol
    assert redis_url == "rediss://redis-server:6379"


def test_get_redis_url_from_environment_with_username_password(monkeypatch):
    """Test when username and password are provided"""
    # Set the environment variables
    monkeypatch.setenv("REDIS_HOST", "redis-server")
    monkeypatch.setenv("REDIS_PORT", "6379")
    monkeypatch.setenv("REDIS_USERNAME", "user")
    monkeypatch.setenv("REDIS_PASSWORD", "password")

    # Call the function to get the Redis URL
    redis_url = get_redis_url_from_environment()

    # Assert that the returned URL includes username:password@
    assert redis_url == "redis://user:password@redis-server:6379"


def test_get_redis_url_from_environment_with_password_only(monkeypatch):
    """Test when only password is provided"""
    # Set the environment variables
    monkeypatch.setenv("REDIS_HOST", "redis-server")
    monkeypatch.setenv("REDIS_PORT", "6379")
    monkeypatch.setenv("REDIS_PASSWORD", "password")
    # Ensure username is not set
    monkeypatch.delenv("REDIS_USERNAME", raising=False)
    monkeypatch.delenv("REDIS_SSL", raising=False)

    # Call the function to get the Redis URL
    redis_url = get_redis_url_from_environment()

    # Assert that the returned URL includes :password@
    assert redis_url == "redis://password@redis-server:6379"


def test_get_redis_url_from_environment_with_all_options(monkeypatch):
    """Test when all options are provided"""
    # Set the environment variables
    monkeypatch.setenv("REDIS_HOST", "redis-server")
    monkeypatch.setenv("REDIS_PORT", "6379")
    monkeypatch.setenv("REDIS_USERNAME", "user")
    monkeypatch.setenv("REDIS_PASSWORD", "password")
    monkeypatch.setenv("REDIS_SSL", "true")

    # Call the function to get the Redis URL
    redis_url = get_redis_url_from_environment()

    # Assert that the returned URL includes all components
    assert redis_url == "rediss://user:password@redis-server:6379"


def test_get_redis_url_from_environment_missing_host_port(monkeypatch):
    """Test error when required variables are missing"""
    # Make sure these environment variables don't exist
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.delenv("REDIS_PORT", raising=False)

    # Call the function and expect a ValueError
    with pytest.raises(ValueError) as excinfo:
        get_redis_url_from_environment()

    # Check the error message
    assert (
        "Either 'REDIS_URL' or both 'REDIS_HOST' and 'REDIS_PORT' must be specified"
        in str(excinfo.value)
    )


def test_get_redis_url_from_environment_missing_port(monkeypatch):
    """Test error when only REDIS_HOST is provided but REDIS_PORT is missing"""
    # Make sure REDIS_URL doesn't exist and set only REDIS_HOST
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_PORT", raising=False)
    monkeypatch.setenv("REDIS_HOST", "redis-server")

    # Call the function and expect a ValueError
    with pytest.raises(ValueError) as excinfo:
        get_redis_url_from_environment()

    # Check the error message
    assert (
        "Either 'REDIS_URL' or both 'REDIS_HOST' and 'REDIS_PORT' must be specified"
        in str(excinfo.value)
    )


def test_max_connections_in_cluster_kwargs():
    """Test that max_connections is included in Redis cluster kwargs"""
    kwargs = _get_redis_cluster_kwargs()
    assert (
        "max_connections" in kwargs
    ), "max_connections should be in available Redis cluster kwargs"


def test_get_redis_async_client_with_connection_pool():
    """Test that connection_pool parameter is properly passed to Redis client"""
    # Create a mock connection pool
    mock_pool = MagicMock(spec=async_redis.BlockingConnectionPool)

    # Mock the Redis client creation
    with (
        patch("litellm._redis.async_redis.Redis") as mock_redis,
        patch("litellm._redis._get_redis_client_logic") as mock_logic,
    ):
        # Configure mock to return basic redis kwargs
        mock_logic.return_value = {"host": "localhost", "port": 6379, "db": 0}

        # Call get_redis_async_client with connection_pool
        get_redis_async_client(connection_pool=mock_pool)

        # Verify Redis was called with connection_pool in kwargs
        call_kwargs = mock_redis.call_args[1]
        assert (
            "connection_pool" in call_kwargs
        ), "connection_pool should be passed to Redis client"
        assert (
            call_kwargs["connection_pool"] == mock_pool
        ), "connection_pool should match the provided pool"


def test_get_redis_async_client_without_connection_pool():
    """Test that Redis client works without connection_pool parameter"""
    with (
        patch("litellm._redis.async_redis.Redis") as mock_redis,
        patch("litellm._redis._get_redis_client_logic") as mock_logic,
    ):
        # Configure mock to return basic redis kwargs
        mock_logic.return_value = {"host": "localhost", "port": 6379, "db": 0}

        # Call get_redis_async_client without connection_pool
        get_redis_async_client()

        # Verify Redis was called without connection_pool in kwargs
        call_kwargs = mock_redis.call_args[1]
        assert (
            "connection_pool" not in call_kwargs
        ), "connection_pool should not be in kwargs when not provided"


def test_init_redis_sentinel_uses_sentinel_kwargs_for_sentinel_auth():
    """Sentinel auth must be passed via `sentinel_kwargs`, not `password`."""
    redis_kwargs = {
        "sentinel_nodes": [("sentinel-1", 26379)],
        "service_name": "mymaster",
        "sentinel_password": "sentinel-pass",
        "password": "redis-pass",
    }

    with patch("litellm._redis.redis.Sentinel") as mock_sentinel:
        mock_sentinel_instance = MagicMock()
        mock_sentinel_instance.master_for.return_value = MagicMock()
        mock_sentinel.return_value = mock_sentinel_instance

        _init_redis_sentinel(redis_kwargs)

        _, kwargs = mock_sentinel.call_args
        assert kwargs["socket_timeout"] == REDIS_SOCKET_TIMEOUT
        assert kwargs["sentinel_kwargs"] == {"password": "sentinel-pass"}
        assert kwargs["password"] == "redis-pass"


def test_init_async_redis_sentinel_uses_sentinel_kwargs_for_sentinel_auth():
    """Async Sentinel auth must be passed via `sentinel_kwargs`, not `password`."""
    redis_kwargs = {
        "sentinel_nodes": [("sentinel-1", 26379)],
        "service_name": "mymaster",
        "sentinel_password": "sentinel-pass",
        "password": "redis-pass",
    }

    with patch("litellm._redis.async_redis.Sentinel") as mock_sentinel:
        mock_sentinel_instance = MagicMock()
        mock_sentinel_instance.master_for.return_value = MagicMock()
        mock_sentinel.return_value = mock_sentinel_instance

        _init_async_redis_sentinel(redis_kwargs)

        _, kwargs = mock_sentinel.call_args
        assert kwargs["socket_timeout"] == REDIS_SOCKET_TIMEOUT
        assert kwargs["sentinel_kwargs"] == {"password": "sentinel-pass"}
        assert kwargs["password"] == "redis-pass"
