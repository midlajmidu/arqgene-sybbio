# ⏱️ SynB Performance, Timings & Troubleshooting Guide

This guide details how the SynB platform manages file uploads, execution times, system boundaries, and server timeouts. Use this document to troubleshoot slow runs, connection drops, and HTTP error codes.

---

## 🔍 1. Upload & Size Boundaries

To protect the server node from memory exhaustion and denial-of-service (DoS) conditions, the backend enforces size boundaries on incoming payloads:

| Boundary Parameter | Value | Location in Code | HTTP Response Code |
| :--- | :--- | :--- | :--- |
| **Max Upload Size (`MAX_UPLOAD_BYTES`)** | 50 MB | [`backend/utils/solve_utils.py`](file:///Users/muhammedmidlaj/Desktop/arqgene-sybbio/backend/utils/solve_utils.py) | `413 Payload Too Large` |
| **FVA Model Size Limit** | 2,000 Reactions | [`backend/exceptions.py`](file:///Users/muhammedmidlaj/Desktop/arqgene-sybbio/backend/exceptions.py) | `400 Bad Request` (`MODEL_TOO_LARGE`) |

### Troubleshooting Upload Errors
*   **HTTP 413 (File too large):** If you upload a massive SBML model exceeding 50 MB, the server immediately drops the payload before parsing. Compress or reduce the model size, or edit `MAX_UPLOAD_BYTES` in [`solve_utils.py`](file:///Users/muhammedmidlaj/Desktop/arqgene-sybbio/backend/utils/solve_utils.py).
*   **HTTP 400 (`MODEL_TOO_LARGE`):** If you try to run Flux Variability Analysis (FVA) on a model containing more than 2,000 reactions, the backend blocks the request. To bypass this check, you must check the confirmation box in the Streamlit UI (which passes `confirm_full_model=True` to the API).

---

## ⏱️ 2. Server Timeout Rules

The FastAPI router limits the wall-clock execution time for every query. Long-running scientific tasks are wrapped in `asyncio.wait_for(...)` to ensure they don't lock worker threads indefinitely:

| Timeout Context | Duration | Error Raised | HTTP Code |
| :--- | :--- | :--- | :--- |
| **Standard Solves (`DEFAULT_SOLVER_TIMEOUT`)** | 60 seconds | `asyncio.TimeoutError` | `408 Request Timeout` |
| **Complex Solves (`VALIDATION_SOLVER_TIMEOUT`)**| 600 seconds | `asyncio.TimeoutError` | `408 Request Timeout` |
| **Metadata Operations (`_MEDIUM_SHORT_TIMEOUT`)** | 30 seconds | `asyncio.TimeoutError` | `408 Request Timeout` |

### Where Timeouts Occur:
*   **`DEFAULT_SOLVER_TIMEOUT` (60s):** Applied to `/upload-model`, `/run-fba`, `/run-pfba`, and `/models/{model_id}/objective`.
*   **`VALIDATION_SOLVER_TIMEOUT` (600s / 10m):** Applied to `/validate-model`, `/models/{model_id}/fva`, and `/models/{model_id}/production-envelope`.
*   **`_MEDIUM_SHORT_TIMEOUT` (30s):** Applied to fetching or updating the growth medium exchanges.

---

## 📈 3. Profiling Execution Time

The platform automatically profiles and logs the time taken by each request down to the millisecond.

### The `X-Process-Time-Ms` Header
The backend includes a custom HTTP middleware ([`backend/main.py`](file:///Users/muhammedmidlaj/Desktop/arqgene-sybbio/backend/main.py)) that measures processing duration:
```python
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"
    return response
```
*   **How to read it:** You can open your browser developer tools (Network tab) and inspect the response headers of any API call. Look for the `X-Process-Time-Ms` key to see exactly how long the backend took to compute the result.

### Server-Side Debug Logs
The backend logs the method, path, status, and duration in `backend.log`:
```text
INFO:     127.0.0.1:49448 - "GET /api/v1/health HTTP/1.1" 200 OK
DEBUG:    POST /api/v1/run-fba → 200 (12.4 ms)
DEBUG:    POST /api/v1/validate-model → 200 (8450.2 ms)
```

---

## ⚡ 4. How to Optimize Performance

If you are running into slow computations or timeout errors, apply these optimization strategies:

1.  **Restrict FVA to a Reaction Subset:**
    *   *Why:* Full FVA runs 2 linear programs per reaction. For a model with 3,000 reactions, this is 6,000 LP solves.
    *   *Solution:* In the **Flux Variability** tab, input a specific list of reaction IDs (one per line, e.g., glycolytic or TCA pathway reactions) in the text box. This restricts the solves only to those reactions, running in seconds instead of minutes.
2.  **Reduce Pareto Scanning Steps:**
    *   *Why:* The Production Envelope Pareto scan runs `steps + 1` sequential LP solves.
    *   *Solution:* Reduce the step count (e.g., from 100 to 20 or 30). This provides a slightly less smooth graph but reduces computation time by 80%.
3.  **Upgrade the LP Solver:**
    *   *Why:* GLPK is a stable, open-source solver, but it is single-threaded and slower on genome-scale models.
    *   *Solution:* Install a commercial-grade solver (Gurobi or CPLEX) or a high-performance open-source alternative like HiGHS (`pip install highs`). Select it from the sidebar dropdown.
4.  **Decrease Validation Overhead:**
    *   *Why:* Running full validation on large models with FVA enabled causes a massive latency spike.
    *   *Solution:* When validating, toggling the FVA check off will run only the topological checks (mass balance, bounds, orphans), making it nearly instantaneous.
