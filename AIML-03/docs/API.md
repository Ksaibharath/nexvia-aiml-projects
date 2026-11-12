# API Documentation

The FastAPI gateway (`src/serving/api.py`) and the cloud serverless handlers (`api/index.mjs` for Vercel and `netlify/functions/api.mjs` for Netlify) expose identical REST JSON endpoints. Interactive Swagger documentation is automatically served at `/docs` when running FastAPI, and the full OpenAPI 3.1 specification is exported to [`docs/openapi.json`](openapi.json).

---

## 1. `GET /health` (alias: `GET /api/health`)
Liveness, active inference backend, catalog size, and Redis/memory cache telemetry.

### Example Request
```bash
curl -s http://localhost:8080/health | python3 -m json.tool
```

### Example Response
```json
{
  "status": "ok",
  "backend": {
    "triton_url": "localhost:8000",
    "triton_live": false,
    "active_engine": "local_exported_weights"
  },
  "cache": {
    "backend": "redis",
    "model_version": "1",
    "hits": 42,
    "misses": 18,
    "hit_rate": 0.7
  },
  "n_catalog_items": 8839,
  "n_warm_users": 4640
}
```

---

## 2. `GET /recommend` (alias: `GET /api/recommend`)
Primary two-stage recommendation endpoint with automatic deterministic A/B routing, versioned Redis caching, and cold-start fallbacks.

### Query Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `user_id` | `int` | **Required** | MovieLens user ID (use e.g. `99999999` for a brand-new cold-start visitor) |
| `k` | `int` | `10` | Number of ranked recommendations to return (`1 .. 50`) |
| `liked` | `string` | `""` | Comma-separated `item_idx` values picked during new-user onboarding (e.g. `"284,263,2016"`) |
| `genres` | `string` | `""` | Comma-separated genre names for cold-start round-robin popularity (e.g. `"Sci-Fi,Action"`) |
| `variant` | `string` | `""` | Optional A/B override (`"A"` = Wide & Deep `ranker_wd`, `"B"` = DCN-V2 `ranker`; empty = MD5 hash) |
| `bypass_cache` | `bool` | `false` | When `true`, skips cache lookup to force live neural inference |

### Routing Paths (`source` field in response)
* `"cache"` — Returned in `<0.1 ms` from versioned cache (`rec:v{model_version}:{variant}:{user_id}:{k}`).
* `"warm"` — Known user with `>= 5` liked items: `UserTower -> FAISS HNSW Top-300 -> Stage-2 Ranker (A or B) Top-K`.
* `"warm_blend"` — Lukewarm user with `1..4` liked items: 50% two-stage neural pipeline + 50% favourite-genre popularity.
* `"content_based"` — New user who supplied `liked` movie IDs: L2-normalized mean item embedding queried against FAISS.
* `"popularity_genre"` — New user who supplied `genres`: round-robin recent popularity across chosen genres.
* `"popularity"` — Brand-new user with zero history: top 90-day popular films.

### Example Request & Response
```bash
curl -s "http://localhost:8080/recommend?user_id=5&k=3"
```
```json
{
  "user_id": 5,
  "variant": "B",
  "ranker_model": "ranker",
  "source": "warm",
  "items": [
    {
      "item_idx": 822,
      "movieId": 1028,
      "title": "Mary Poppins (1964)",
      "year": 1964,
      "genres": "Children|Comedy|Fantasy|Musical",
      "avg_rating": 3.81,
      "n_pos": 254,
      "score": 2.062,
      "retrieval_score": 0.2284
    }
  ],
  "history_preview": [
    {
      "item_idx": 1099,
      "movieId": 1380,
      "title": "Grease (1978)",
      "year": 1978,
      "genres": "Comedy|Musical|Romance",
      "avg_rating": 3.38,
      "n_pos": 193
    }
  ],
  "latency_ms": 2.41
}
```

---

## 3. Supporting Interactive UI Endpoints
* `GET /api/compare?user_id=5&k=10` — Runs Variant A (`Wide & Deep`) and Variant B (`DCN-V2`) side-by-side on the same candidate set and reports Top-K overlap.
* `GET /api/users` — Lists sample warm & lukewarm MovieLens user profiles with their top genres and recent liked titles.
* `GET /api/items?q=Matrix&genre=Sci-Fi&limit=40` — Searches catalog items and lists all available genres for onboarding.
* `GET /api/metrics` — Returns offline retrieval/ranking metrics, FAISS benchmarks, temporal split summary, and live cache hit rates.
* `GET /api/ab-results` — Returns counterfactual replay A/B test summary (`SRM p-value`, Welch's t-test, 95% bootstrap CIs, latency benchmarks).
