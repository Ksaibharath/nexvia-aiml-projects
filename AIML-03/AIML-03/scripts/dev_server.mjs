/**
 * Zero-dependency local development server (Node.js 18+).
 * Run `npm start` or `node scripts/dev_server.mjs` to launch the full Web UI + API locally in <1 second!
 */
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { routeApiRequest } from "../lib/recsysEngine.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = path.resolve(__dirname, "../public");
const PORT = parseInt(process.env.PORT || "8080", 10);

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
};

const server = http.createServer((req, res) => {
  const urlObj = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
  const pathname = urlObj.pathname;

  if (pathname === "/health" || pathname === "/recommend" || pathname.startsWith("/api/")) {
    try {
      const queryObj = Object.fromEntries(urlObj.searchParams.entries());
      const payload = routeApiRequest(pathname, queryObj);
      res.writeHead(200, { "Content-Type": "application/json; charset=utf-8", "Access-Control-Allow-Origin": "*" });
      res.end(JSON.stringify(payload));
    } catch (err) {
      res.writeHead(500, { "Content-Type": "application/json; charset=utf-8" });
      res.end(JSON.stringify({ error: String(err.message || err) }));
    }
    return;
  }

  let filePath = path.join(PUBLIC_DIR, pathname === "/" ? "index.html" : pathname);
  if (!filePath.startsWith(PUBLIC_DIR) || !fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
    filePath = path.join(PUBLIC_DIR, "index.html");
  }
  const ext = path.extname(filePath);
  res.writeHead(200, { "Content-Type": MIME[ext] || "application/octet-stream" });
  fs.createReadStream(filePath).pipe(res);
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`\n=============================================================`);
  console.log(`  Two-Stage Recommendation Engine running locally!`);
  console.log(`  -> Open Web Dashboard:  http://localhost:${PORT}`);
  console.log(`  -> Health Endpoint:     http://localhost:${PORT}/health`);
  console.log(`  -> Sample Recommend:    http://localhost:${PORT}/recommend?user_id=5&k=10`);
  console.log(`=============================================================\n`);
});
