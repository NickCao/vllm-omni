import pytest

from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector


@pytest.mark.parametrize(
    "threshold, size",
    [
        (100, 50),  # size < threshold, shm
        (100, 200),  # size > threshold, inline
    ],
)
def test_shared_memory_connector(threshold, size):
    from_stage = "dummy_from_stage"
    to_stage = "dummy_to_stage"
    key = "dummy_key"
    data = b" " * size

    tx = SharedMemoryConnector({"threshold": threshold})
    rx = SharedMemoryConnector({"threshold": threshold})

    success, _, metadata = tx.put(
        from_stage=from_stage,
        to_stage=to_stage,
        put_key=key,
        data=data,
    )
    assert success

    rx_data, _ = rx.get(
        from_stage=from_stage,
        to_stage=to_stage,
        get_key=key,
        metadata=metadata,
    )
    assert rx_data == data
