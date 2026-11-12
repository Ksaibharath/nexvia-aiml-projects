import { routeApiRequest } from '../../lib/recsysEngine.mjs';
export async function handler(event) {
  try {
    let rawPath = event.path || '/';
    if ((rawPath === '/.netlify/functions/api' || rawPath === '/') && event.rawUrl) { try { rawPath = new URL(event.rawUrl).pathname; } catch (_e) {} }
    const payload = routeApiRequest(rawPath, event.queryStringParameters || {});
    return { statusCode: 200, headers: { 'Content-Type': 'application/json; charset=utf-8', 'Access-Control-Allow-Origin': '*' }, body: JSON.stringify(payload) };
  } catch (err) { return { statusCode: 500, headers: { 'Content-Type': 'application/json; charset=utf-8' }, body: JSON.stringify({ error: String(err.message || err) }) }; }
}
