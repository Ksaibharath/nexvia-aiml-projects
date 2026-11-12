# Deploy: GitHub → Vercel / Netlify

## Option A — Vercel (recommended, Next.js)
1. Push repo to GitHub (see root README).
2. vercel.com → Add New → Project → import fraud-platform.
3. Framework preset: Next.js. Build: npm run build. No env vars needed.
4. Deploy → live URL. API routes: /api/score, /api/stream, /api/metrics, /api/drift, /api/health.

## Option B — Netlify
1. app.netlify.com → Add new site → Import from Git.
2. Build command: npm run build · Publish: .next (netlify.toml included).
3. Deploy.

## Full MLOps stack (optional, local/K8s)
See backend/README.md: docker compose (Kafka/Redis/Postgres/MLflow) + Minikube manifests in backend/k8s/.
Vercel/Netlify cannot run Kafka/Postgres/Feast — the Next.js app is the deployable twin with the same scoring contract.
