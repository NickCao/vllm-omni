# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Any

import ray
from ray import ObjectRef

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)


@ray.remote
class RayRefStore:
    """Named Ray actor that maps string keys to ObjectRefs.

    Acts as a distributed hash map backed by the Ray object store.
    The sender puts data into the object store via ``ray.put()`` and
    registers the resulting ``ObjectRef`` under a key in this actor.
    The receiver looks up the ``ObjectRef`` by key and calls
    ``ray.get()`` to fetch the data — Ray handles cross-node transfer
    transparently.
    """

    def __init__(self):
        self._store: dict[str, ObjectRef] = {}

    def put(self, key: str, ref: list[ObjectRef]) -> None:
        """Register the ``ObjectRef`` in *ref[0]* under *key*.

        The ref is wrapped in a list to prevent Ray from automatically
        resolving the ``ObjectRef`` when passing it to this remote
        method.
        """
        self._store[key] = ref[0]

    def get(self, key: str) -> ObjectRef | None:
        """Return the ``ObjectRef`` for *key*, or ``None``."""
        return self._store.get(key)

    def delete(self, keys: list[str]) -> list[ObjectRef]:
        """Remove *keys* and return the freed ``ObjectRef`` s."""
        freed: list[ObjectRef] = []
        for k in keys:
            ref = self._store.pop(k, None)
            if ref is not None:
                freed.append(ref)
        return freed

    def delete_by_prefix(self, prefix: str) -> list[ObjectRef]:
        """Remove all keys that start with *prefix*."""
        to_delete = [k for k in self._store if k.startswith(prefix)]
        return self.delete(to_delete)

    def keys(self) -> list[str]:
        """List all registered keys (debugging / health checks)."""
        return list(self._store.keys())


class RayConnector(OmniConnectorBase):
    """Connector that uses the Ray object store for data transfer.

    This connector requires no infrastructure beyond a running Ray
    cluster.  It stores data via ``ray.put()`` and registers the
    resulting ``ObjectRef`` in a :class:`RayRefStore` named actor so
    that receivers on any node can retrieve them with ``ray.get()``.
    Cross-node data movement is handled transparently by Ray,
    including zero-copy via Ray Direct Transport (RDT) on
    RDMA-equipped clusters.

    Data is passed to ``ray.put()`` without manual serialization —
    Ray handles serialization natively, which enables the RDT fast
    path for tensors when ``rdt`` is enabled.

    Configuration keys (all optional):
        actor_name         (str)       – Name of the ``RayRefStore`` actor.
                                         Default ``"vllm_omni_ray_ref_store"``.
        tensor_transport   (str|None)  – Transport hint passed to ``ray.put()``
                                         (e.g. ``"nixl"`` for RDT).  Default ``None``.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._actor_name: str = config.get(
            "actor_name", "vllm_omni_ray_ref_store"
        )
        self._tensor_transport: str | None = config.get("tensor_transport", None)

        self._metrics: dict[str, int] = {
            "puts": 0,
            "gets": 0,
            "errors": 0,
            "timeouts": 0,
        }

        self._closed = False
        self._ref_store: ray.actor.ActorHandle | None = None

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _ensure_ref_store(self) -> None:
        """Lazily obtain or create the ``RayRefStore`` actor.

        Deferred from ``__init__`` because the connector may be
        created before Ray is initialized (orchestrator process) or
        in a subprocess that is not a Ray actor (vLLM's EngineCore
        spawned by ``multiproc_executor``).  In either case we
        connect to the existing cluster via ``address="auto"``.
        """
        if self._ref_store is not None:
            return

        if not ray.is_initialized():
            ray.init(address="auto", ignore_reinit_error=True)

        # get_if_exists=True atomically creates the actor if it does
        # not exist, or returns a handle to the existing one.
        self._ref_store = RayRefStore.options(  # type: ignore[attr-defined]
            name=self._actor_name,
            get_if_exists=True,
        ).remote()
        logger.info(
            "RayConnector: obtained RayRefStore actor '%s'",
            self._actor_name,
        )

    # ------------------------------------------------------------------
    # OmniConnectorBase interface
    # ------------------------------------------------------------------

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        try:
            self._ensure_ref_store()
            key = self._make_key(put_key, from_stage, to_stage)

            # Let Ray handle serialization natively.  When a tensor
            # transport is configured (e.g. "nixl"), tensors are
            # transferred via RDT zero-copy.
            obj_ref = ray.put(
                data, _tensor_transport=self._tensor_transport
            )

            # Wrap ref in a list so Ray does not auto-resolve it
            # when passing to the actor's remote method.  We wait for
            # the actor call to confirm the ref is registered — if it
            # fails, the local obj_ref goes out of scope and Ray GCs
            # the object, so there is no leak.
            ray.get(self._ref_store.put.remote(key, [obj_ref]))

            self._metrics["puts"] += 1

            logger.debug(
                "RayConnector: stored %s (%s -> %s)",
                key,
                from_stage,
                to_stage,
            )
            return True, 0, None

        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("RayConnector put failed: %s", e)
            return False, 0, None

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, int] | None:
        self._ensure_ref_store()
        retries = 20
        sleep_s = 0.05
        key = self._make_key(get_key, from_stage, to_stage)

        t0 = time.perf_counter()

        for attempt in range(retries):
            try:
                # Ask the actor for the ObjectRef.
                obj_ref = ray.get(self._ref_store.get.remote(key))
                if obj_ref is not None:
                    t_fetch_start = time.perf_counter()
                    data = ray.get(obj_ref)
                    t_fetch_end = time.perf_counter()

                    fetch_ms = (t_fetch_end - t_fetch_start) * 1000

                    self._metrics["gets"] += 1
                    total_ms = (t_fetch_end - t0) * 1000
                    logger.info(
                        "[RAY GET] %s: fetch=%.1fms, total=%.1fms",
                        get_key,
                        fetch_ms,
                        total_ms,
                    )
                    return data, 0

            except Exception as e:
                logger.debug(
                    "RayConnector get attempt %s failed: %s", attempt, e
                )

            if attempt < retries - 1:
                time.sleep(sleep_s)

        self._metrics["timeouts"] += 1
        logger.warning("RayConnector: timeout waiting for %s", key)
        return None

    def cleanup(self, request_id: str) -> None:
        """Delete all refs whose key starts with *request_id*."""
        if self._ref_store is None:
            return
        try:
            ray.get(self._ref_store.delete_by_prefix.remote(request_id))
            logger.debug(
                "RayConnector: cleanup completed for %s", request_id
            )
        except Exception as e:
            logger.debug("RayConnector: cleanup error for %s: %s", request_id, e)

    def health(self) -> dict[str, Any]:
        if self._closed:
            return {"status": "unhealthy", "error": "connector is closed"}
        try:
            self._ensure_ref_store()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
        try:
            num_keys = len(ray.get(self._ref_store.keys.remote()))
        except Exception:
            num_keys = -1
        return {
            "status": "healthy",
            "actor_name": self._actor_name,
            "tensor_transport": self._tensor_transport,
            "stored_keys": num_keys,
            **self._metrics,
        }

    def close(self) -> None:
        """Release the actor handle."""
        if self._closed:
            return
        self._closed = True
        self._ref_store = None
        logger.info("RayConnector closed")
