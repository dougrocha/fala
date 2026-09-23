# Notes

## Next: distributed setup (router + workers)

Run several copies of the inference server as workers, and put a small Odin router in front.

```
client
  ↓
router      :8080
  ↓
worker A    :8081
worker B    :8082
worker C    :8083
```

First exercise: 2 workers on different ports, and a router that sends each new request to the worker with the fewest active requests.

Topics this covers:
- service discovery / worker registration
- load balancing
- health checks
- request routing
- retries and failure handling
- backpressure
- per-worker queue depth
- sticky routing when state matters (prompt cache)
- metrics across nodes

Plan:
- One binary with 2 modes: `worker <port>` (current server) and `router <port> <worker ports...>` (no model).
- The router forwards raw bytes both ways, so SSE streams pass through unchanged.
- Track active requests per worker with an atomic counter.
- Add a `GET /health` endpoint on the worker for health checks.

Warning for this machine (RTX 3090, 24 GB):
- Each worker loads its own model copy: about 2.5 GB (1.7 GB weights + 448 MiB KV + 300 MiB compute). 3 workers fit.
- All workers share 1 GPU and its memory bandwidth. Each worker reads the weights on its own, so 2 workers with 4 slots each should give **less** total tok/s than 1 worker with 8 slots (estimate). Measure this.
- So on 1 GPU this is a learning exercise for the distributed parts, not a speed-up. Real gains need more GPUs or more machines.
