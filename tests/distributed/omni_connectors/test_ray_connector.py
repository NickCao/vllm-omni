# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import uuid

import pytest
import ray

from vllm_omni.distributed.omni_connectors.connectors.ray_connector import (
    RayConnector,
)
from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec


@pytest.fixture(scope="module")
def ray_init():
    """Start a local Ray cluster for the test module."""
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def connector(ray_init):
    """Create a RayConnector with a unique actor name per test."""
    config = {"actor_name": f"test_ray_ref_store_{uuid.uuid4().hex[:8]}"}
    conn = RayConnector(config)
    yield conn
    conn.close()


# -- Factory ---------------------------------------------------------


def test_create_ray_connector_via_factory(ray_init):
    """RayConnector can be created through OmniConnectorFactory."""
    spec = ConnectorSpec(
        name="RayConnector",
        extra={"actor_name": f"test_factory_{uuid.uuid4().hex[:8]}"},
    )
    conn = OmniConnectorFactory.create_connector(spec)
    assert isinstance(conn, RayConnector)
    conn.close()


# -- Put / Get -------------------------------------------------------


def test_put_get_dict(connector):
    """Round-trip a plain dict through put/get."""
    data = {"key": "value", "list": [1, 2, 3]}

    success, size, metadata = connector.put(
        "stage_0", "stage_1", "req_dict", data
    )
    assert success is True
    assert metadata is None

    result = connector.get("stage_0", "stage_1", "req_dict")
    assert result is not None
    retrieved, ret_size = result
    assert retrieved == data


def test_put_get_tensor(connector):
    """Round-trip a torch.Tensor through put/get."""
    import torch

    tensor = torch.rand(64, 64, dtype=torch.float32)

    success, _, _ = connector.put(
        "stage_0", "stage_1", "req_tensor", {"tensor": tensor}
    )
    assert success is True

    result = connector.get("stage_0", "stage_1", "req_tensor")
    assert result is not None
    retrieved, _ = result
    assert torch.equal(tensor, retrieved["tensor"])


def test_put_get_numpy(connector):
    """Round-trip a numpy array through put/get."""
    import numpy as np

    arr = np.array([[1, 2, 3], [4, 5, 6]])

    success, _, _ = connector.put(
        "stage_0", "stage_1", "req_numpy", {"array": arr}
    )
    assert success is True

    result = connector.get("stage_0", "stage_1", "req_numpy")
    assert result is not None
    retrieved, _ = result
    assert np.array_equal(arr, retrieved["array"])


def test_get_missing_key(connector):
    """get() returns None for a key that was never put."""
    result = connector.get("stage_0", "stage_1", "nonexistent")
    assert result is None


def test_put_get_multiple_keys(connector):
    """Multiple independent keys can be stored and retrieved."""
    for i in range(5):
        data = {"index": i}
        success, _, _ = connector.put(
            "stage_0", "stage_1", f"req_multi_{i}", data
        )
        assert success is True

    for i in range(5):
        result = connector.get("stage_0", "stage_1", f"req_multi_{i}")
        assert result is not None
        retrieved, _ = result
        assert retrieved == {"index": i}


def test_put_get_different_stages(connector):
    """Keys are scoped by stage pair — same put_key, different stages."""
    data_a = {"from": "0_to_1"}
    data_b = {"from": "1_to_2"}

    connector.put("stage_0", "stage_1", "req_stage", data_a)
    connector.put("stage_1", "stage_2", "req_stage", data_b)

    result_a = connector.get("stage_0", "stage_1", "req_stage")
    result_b = connector.get("stage_1", "stage_2", "req_stage")

    assert result_a is not None
    assert result_b is not None
    assert result_a[0] == data_a
    assert result_b[0] == data_b


# -- Cleanup ---------------------------------------------------------


def test_cleanup_removes_keys(connector):
    """cleanup() removes keys matching the request-id prefix."""
    connector.put("stage_0", "stage_1", "cleanup_test", {"a": 1})

    # Key should be retrievable before cleanup.
    assert connector.get("stage_0", "stage_1", "cleanup_test") is not None

    connector.cleanup("cleanup_test")

    # After cleanup, the key should be gone.
    assert connector.get("stage_0", "stage_1", "cleanup_test") is None


# -- Health ----------------------------------------------------------


def test_health_reports_status(connector):
    """health() returns a dict with expected fields."""
    health = connector.health()
    assert health["status"] == "healthy"
    assert "actor_name" in health
    assert "stored_keys" in health
    assert isinstance(health["stored_keys"], int)


# -- Close -----------------------------------------------------------


def test_close_is_idempotent(connector):
    """close() can be called multiple times without error."""
    connector.close()
    connector.close()


def test_health_after_close(connector):
    """health() reports unhealthy after close."""
    connector.close()
    health = connector.health()
    assert health["status"] == "unhealthy"
