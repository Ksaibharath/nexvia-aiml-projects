import asyncio, time, httpx, numpy as np, pandas as pd
URL = "http://localhost:8080"
te = pd.read_parquet("data/processed/labels.parquet").query("split == 'test'").sample(3000, random_state=0)
async def main(conc=20):
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient(timeout=10) as c:
        async def one(r):
            async with sem:
                t = time.perf_counter()
                resp = await c.post(f"{URL}/score", json={"transaction_id": int(r.transaction_id), "uid": r.uid})
                return (time.perf_counter() - t) * 1000, resp.status_code
        t0 = time.perf_counter(); res = await asyncio.gather(*[one(r) for r in te.itertuples()]); dt = time.perf_counter() - t0
    lat = np.array([x[0] for x in res]); ok = sum(x[1] == 200 for x in res)
    print(f"concurrency {conc}: ok {ok}/{len(res)} | p50 {np.percentile(lat, 50):.1f} ms | p95 {np.percentile(lat, 95):.1f} ms | p99 {np.percentile(lat, 99):.1f} ms | {len(res) / dt:.0f} req/s")
for conc in (1, 10, 50): asyncio.run(main(conc))
