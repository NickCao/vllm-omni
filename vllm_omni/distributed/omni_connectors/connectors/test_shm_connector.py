import pytest

from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector


@pytest.mark.parametrize(
    "size, threshold",
    [
        (50, 100),  # size < threshold, inline
        (200, 100),  # size > threshold, shm
    ],
)
def test_shared_memory_connector(size, threshold):
    from_stage = "dummy_from_stage"
    to_stage = "dummy_to_stage"
    key = "dummy_key"
    data = b" " * size

    tx = SharedMemoryConnector({"shm_threshold_bytes": threshold})
    rx = SharedMemoryConnector({"shm_threshold_bytes": threshold})

    success, _, metadata = tx.put(
        from_stage=from_stage,
        to_stage=to_stage,
        put_key=key,
        data=data,
    )
    assert success

    if size < threshold:
        assert "inline_bytes" in metadata
    else:
        assert "shm" in metadata

    rx_data, _ = rx.get(
        from_stage=from_stage,
        to_stage=to_stage,
        get_key=key,
        metadata=metadata,
    )
    assert rx_data == data
