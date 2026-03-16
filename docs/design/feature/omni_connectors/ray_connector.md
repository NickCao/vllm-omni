# RayConnector

## When to Use

Best for multi-node deployments that already run on a Ray cluster.  No
additional services (Mooncake, etcd, etc.) are required — the connector
uses the Ray object store for data transfer and a named Ray actor as the
distributed key-value index.

On RDMA-equipped clusters, enable
[Ray Direct Transport (RDT)](https://docs.ray.io/en/latest/ray-core/direct-transport.html)
for performance comparable to native RDMA connectors.

## How It Works

1. **Sender** stores data via `ray.put()` (Ray handles serialization
   natively) and registers the resulting `ObjectRef` in a `RayRefStore`
   named actor.
2. **Receiver** looks up the `ObjectRef` by key from the actor and calls
   `ray.get()` to fetch the data.  Ray handles cross-node transfer
   transparently.

## Configuration

```yaml
runtime:
  connectors:
    connector_of_ray:
      name: RayConnector
      extra:
        actor_name: "vllm_omni_ray_ref_store"  # Named actor identity (default)
        tensor_transport: null         # Transport hint for ray.put() (default: null)
```

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `actor_name` | `"vllm_omni_ray_ref_store"` | Name of the `RayRefStore` actor.  Override for multi-pipeline isolation. |
| `tensor_transport` | `null` | Transport hint passed to `ray.put()` (e.g. `"nixl"` for RDT zero-copy tensor transfer on RDMA-equipped clusters). |

## Notes

- The `RayRefStore` actor is owned by the process whose connector
  creates it first.  It is automatically cleaned up by Ray when that
  process exits.
- `cleanup()` deletes all keys matching the given request-id prefix from
  the actor, allowing the Ray object store to garbage-collect the
  underlying objects.
