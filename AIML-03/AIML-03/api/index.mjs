import { routeApiRequest } from "../lib/recsysEngine.mjs";

export default function handler(req, res) {
  try {
    const urlObj = new URL(req.url, `http://${req.headers.host || "localhost"}`);
    const queryObj = Object.fromEntries(urlObj.searchParams.entries());
    const pathname = queryObj.__path ? `/${queryObj.__path}` : urlObj.pathname;
    const payload = routeApiRequest(pathname, queryObj);
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.statusCode = 200;
    res.end(JSON.stringify(payload));
  } catch (err) {
    res.statusCode = 500;
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end(JSON.stringify({ error: String(err.message || err) }));
  }
}
