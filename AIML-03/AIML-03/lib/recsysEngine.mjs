/**
 * Zero-dependency Serverless Two-Stage Recommendation Engine (Vercel + Netlify)
 * Executes real exported UserTower, ANN Cosine Retrieval, DCN-V2 & Wide-and-Deep Ranking,
 * Cold-Start Fallbacks, MD5 A/B Routing, and Versioned TTL Caching from deploy_bundle/bundle.json.
 */
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

let _state = null;
const _cache = new Map();
let _cacheHits = 0;
let _cacheMisses = 0;

function halfToFloat(h) {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) {
    return (s ? -1 : 1) * Math.pow(2, -14) * (f / Math.pow(2, 10));
  }
  if (e === 0x1f) {
    return f ? NaN : (s ? -Infinity : Infinity);
  }
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / Math.pow(2, 10));
}

function decodeTensor(spec) {
  const buf = Buffer.from(spec.b64, "base64");
  if (spec.dtype === "float32") {
    const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    return { shape: spec.shape, data: new Float32Array(ab) };
  }
  const u16 = new Uint16Array(buf.buffer, buf.byteOffset, buf.byteLength / 2);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    out[i] = halfToFloat(u16[i]);
  }
  return { shape: spec.shape, data: out };
}

function matVecMul(vec, weightTensor, biasArr, relu = false) {
  const [inDim, outDim] = weightTensor.shape;
  const W = weightTensor.data;
  const out = new Float32Array(outDim);
  for (let j = 0; j < outDim; j++) {
    out[j] = biasArr ? biasArr[j] : 0.0;
  }
  for (let i = 0; i < inDim; i++) {
    const vi = vec[i];
    if (vi === 0) continue;
    const rowOff = i * outDim;
    for (let j = 0; j < outDim; j++) {
      out[j] += vi * W[rowOff + j];
    }
  }
  if (relu) {
    for (let j = 0; j < outDim; j++) {
      if (out[j] < 0) out[j] = 0;
    }
  }
  return out;
}

function l2Normalize(vec) {
  let sumSq = 0.0;
  for (let i = 0; i < vec.length; i++) sumSq += vec[i] * vec[i];
  const inv = 1.0 / Math.max(Math.sqrt(sumSq), 1e-12);
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] * inv;
  return out;
}

function loadBundle() {
  if (_state) return _state;
  const candidates = [
    path.resolve(process.cwd(), "deploy_bundle/bundle.json"),
    path.resolve(process.cwd(), "AIML-03/deploy_bundle/bundle.json"),
    path.resolve(process.cwd(), "ALML-03/deploy_bundle/bundle.json"),
  ];
  try {
    if (typeof import.meta !== "undefined" && import.meta && import.meta.url) {
      const dir = path.dirname(fileURLToPath(import.meta.url));
      candidates.push(path.resolve(dir, "../deploy_bundle/bundle.json"));
      candidates.push(path.resolve(dir, "../../deploy_bundle/bundle.json"));
    }
  } catch (_e) {}
  try {
    if (typeof __dirname !== "undefined") {
      candidates.push(path.resolve(__dirname, "../deploy_bundle/bundle.json"));
      candidates.push(path.resolve(__dirname, "../../deploy_bundle/bundle.json"));
      candidates.push(path.resolve(__dirname, "deploy_bundle/bundle.json"));
    }
  } catch (_e) {}
  const bundlePath = candidates.find((p) => fs.existsSync(p));
  if (!bundlePath) {
    throw new Error("deploy_bundle/bundle.json not found");
  }
  const raw = JSON.parse(fs.readFileSync(bundlePath, "utf-8"));

  const itemByIdx = new Map();
  const itemPosByIdx = new Map(); // 0-based position in sub_items, +1 for embedding slot
  const genreSet = new Set();
  raw.items.forEach((it, pos) => {
    itemByIdx.set(it.item_idx, it);
    itemPosByIdx.set(it.item_idx, pos);
    it.genres.split("|").forEach((g) => {
      if (g && g !== "(no genres listed)") genreSet.add(g);
    });
  });

  const sortedRecent = [...raw.items].sort((a, b) => (b.n_pos_recent - a.n_pos_recent) || (b.n_pos - a.n_pos));
  const globalTop = sortedRecent.slice(0, 200).map((x) => x.item_idx);

  const ut = {
    user_emb: decodeTensor(raw.user_tower.user_emb),
    hist_emb: decodeTensor(raw.user_tower.hist_emb),
    mlp_w0: decodeTensor(raw.user_tower.mlp_w0),
    mlp_b0: decodeTensor(raw.user_tower.mlp_b0).data,
    mlp_w1: decodeTensor(raw.user_tower.mlp_w1),
    mlp_b1: decodeTensor(raw.user_tower.mlp_b1).data,
  };

  const wd = {
    ue: decodeTensor(raw.ranker_wd.ue),
    ie: decodeTensor(raw.ranker_wd.ie),
    bn_scale: decodeTensor(raw.ranker_wd.bn_scale).data,
    bn_offset: decodeTensor(raw.ranker_wd.bn_offset).data,
    wide_w: decodeTensor(raw.ranker_wd.wide_w),
    wide_b: decodeTensor(raw.ranker_wd.wide_b).data,
    deep: raw.ranker_wd.deep.map((l) => ({ w: decodeTensor(l.w), b: decodeTensor(l.b).data })),
  };

  const dcn = {
    ue: decodeTensor(raw.ranker_dcn.ue),
    ie: decodeTensor(raw.ranker_dcn.ie),
    bn_scale: decodeTensor(raw.ranker_dcn.bn_scale).data,
    bn_offset: decodeTensor(raw.ranker_dcn.bn_offset).data,
    cross: raw.ranker_dcn.cross.map((l) => ({ w: decodeTensor(l.w), b: decodeTensor(l.b).data })),
    deep: raw.ranker_dcn.deep.map((l) => ({ w: decodeTensor(l.w), b: decodeTensor(l.b).data })),
    out_w: decodeTensor(raw.ranker_dcn.out_w),
    out_b: decodeTensor(raw.ranker_dcn.out_b).data,
  };

  _state = {
    cfg: raw.config,
    nGenres: raw.n_genres,
    nTotalCatalog: raw.n_total_catalog,
    nTotalWarmUsers: raw.n_total_warm_users,
    items: raw.items,
    itemByIdx,
    itemPosByIdx,
    genres: [...genreSet].sort(),
    sortedRecent,
    globalTop,
    users: raw.users,
    itemEmb: decodeTensor(raw.item_embeddings),
    IF: decodeTensor(raw.IF),
    UF: decodeTensor(raw.UF),
    ut,
    wd,
    dcn,
    telemetry: raw.telemetry,
  };
  return _state;
}

export function variantFor(userId, salt = "exp1") {
  const hex = crypto.createHash("md5").update(`${salt}:${userId}`).digest("hex");
  const lastByte = parseInt(hex.slice(-2), 16);
  return lastByte % 2 === 0 ? "A" : "B";
}

function getRow(tensor, rowIdx) {
  const cols = tensor.shape[1];
  const off = rowIdx * cols;
  return tensor.data.subarray(off, off + cols);
}

function runUserTower(st, uSlot, historyItemIdxs) {
  const dim = st.ut.user_emb.shape[1];
  const uVec = getRow(st.ut.user_emb, uSlot);
  const hMean = new Float32Array(dim);
  const recent = historyItemIdxs.slice(-20);
  let count = 0;
  for (const itemIdx of recent) {
    const pos = st.itemPosByIdx.get(itemIdx);
    if (pos !== undefined) {
      const hRow = getRow(st.ut.hist_emb, pos + 1);
      for (let d = 0; d < dim; d++) hMean[d] += hRow[d];
      count++;
    }
  }
  if (count > 0) {
    for (let d = 0; d < dim; d++) hMean[d] /= count;
  }
  const concat = new Float32Array(dim * 2);
  concat.set(uVec, 0);
  concat.set(hMean, dim);
  const h1 = matVecMul(concat, st.ut.mlp_w0, st.ut.mlp_b0, true);
  const out = matVecMul(h1, st.ut.mlp_w1, st.ut.mlp_b1, false);
  return l2Normalize(out);
}

function scoreRankerSingle(st, variant, uSlot, iPos, uFeat, iFeat, retScore, histLen) {
  const nG = st.nGenres;
  const uLen = uFeat.length;
  const iLen = iFeat.length;
  let crossGenre = 0.0;
  for (let g = 0; g < nG; g++) {
    crossGenre += uFeat[uLen - nG + g] * iFeat[iLen - nG + g];
  }

  const denseLen = uLen + iLen + 3;
  const dense = new Float32Array(denseLen);
  dense.set(uFeat, 0);
  dense.set(iFeat, uLen);
  dense[uLen + iLen] = retScore;
  dense[uLen + iLen + 1] = Math.log1p(histLen);
  dense[uLen + iLen + 2] = crossGenre;

  const model = variant === "A" ? st.wd : st.dcn;
  const dNorm = new Float32Array(denseLen);
  for (let j = 0; j < denseLen; j++) {
    dNorm[j] = dense[j] * model.bn_scale[j] + model.bn_offset[j];
  }

  const uEmb = getRow(model.ue, uSlot);
  const iEmb = getRow(model.ie, iPos + 1);
  const x0 = new Float32Array(uEmb.length + iEmb.length + denseLen);
  x0.set(uEmb, 0);
  x0.set(iEmb, uEmb.length);
  x0.set(dNorm, uEmb.length + iEmb.length);

  if (variant === "A") {
    const wideLogit = matVecMul(dNorm, model.wide_w, model.wide_b, false)[0];
    let h = x0;
    for (let li = 0; li < model.deep.length; li++) {
      const isLast = li === model.deep.length - 1;
      h = matVecMul(h, model.deep[li].w, model.deep[li].b, !isLast);
    }
    return wideLogit + h[0];
  } else {
    let x = x0;
    for (let ci = 0; ci < model.cross.length; ci++) {
      const proj = matVecMul(x, model.cross[ci].w, model.cross[ci].b, false);
      const nextX = new Float32Array(x0.length);
      for (let j = 0; j < x0.length; j++) nextX[j] = x0[j] * proj[j] + x[j];
      x = nextX;
    }
    let h = x0;
    for (let li = 0; li < model.deep.length; li++) {
      h = matVecMul(h, model.deep[li].w, model.deep[li].b, true);
    }
    const cat = new Float32Array(x.length + h.length);
    cat.set(x, 0);
    cat.set(h, x.length);
    return matVecMul(cat, model.out_w, model.out_b, false)[0];
  }
}

function popularFallback(st, k, genres = [], excludeSet = new Set()) {
  if (!genres.length) {
    return st.globalTop.filter((id) => !excludeSet.has(id)).slice(0, k).map((id) => ({ ...st.itemByIdx.get(id) }));
  }
  const lists = genres.map((g) =>
    st.sortedRecent
      .filter((it) => it.genres.toLowerCase().includes(g.toLowerCase()))
      .slice(0, 100)
      .map((it) => it.item_idx)
  );
  const out = [];
  const used = new Set(excludeSet);
  let i = 0;
  while (out.length < k && lists.some((l) => i < l.length)) {
    for (const l of lists) {
      if (i < l.length && !used.has(l[i])) {
        used.add(l[i]);
        out.push({ ...st.itemByIdx.get(l[i]) });
        if (out.length >= k) break;
      }
    }
    i++;
  }
  return out;
}

function contentBasedFromLiked(st, likedIds, k) {
  const validPos = likedIds.map((id) => st.itemPosByIdx.get(id)).filter((p) => p !== undefined);
  if (!validPos.length) return popularFallback(st, k);
  const dim = st.itemEmb.shape[1];
  const meanVec = new Float32Array(dim);
  for (const pos of validPos) {
    const row = getRow(st.itemEmb, pos);
    for (let d = 0; d < dim; d++) meanVec[d] += row[d];
  }
  const q = l2Normalize(meanVec);
  const excl = new Set(likedIds);
  const scored = [];
  for (let pos = 0; pos < st.items.length; pos++) {
    const item = st.items[pos];
    if (excl.has(item.item_idx)) continue;
    const v = getRow(st.itemEmb, pos);
    let dot = 0.0;
    for (let d = 0; d < dim; d++) dot += q[d] * v[d];
    scored.push({ item, score: dot });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k).map((s) => ({ ...s.item, retrieval_score: Number(s.score.toFixed(4)) }));
}

function warmRecommend(st, uProfile, variant, k) {
  const q = runUserTower(st, uProfile.slot, uProfile.history);
  const dim = st.itemEmb.shape[1];
  const seen = new Set(uProfile.seen_items || uProfile.history);

  const candidates = [];
  for (let pos = 0; pos < st.items.length; pos++) {
    const it = st.items[pos];
    if (seen.has(it.item_idx)) continue;
    const v = getRow(st.itemEmb, pos);
    let dot = 0.0;
    for (let d = 0; d < dim; d++) dot += q[d] * v[d];
    candidates.push({ pos, item: it, retScore: dot });
  }
  candidates.sort((a, b) => b.retScore - a.retScore);
  const topCand = candidates.slice(0, st.cfg.retrieval_k || 300);

  const uFeat = getRow(st.UF, uProfile.slot);
  const ranked = topCand.map((c) => {
    const iFeat = getRow(st.IF, c.pos);
    const logit = scoreRankerSingle(st, variant, uProfile.slot, c.pos, uFeat, iFeat, c.retScore, uProfile.history.length);
    return {
      ...c.item,
      score: Number(logit.toFixed(4)),
      retrieval_score: Number(c.retScore.toFixed(4)),
    };
  });
  ranked.sort((a, b) => b.score - a.score);
  return ranked.slice(0, k);
}

export function handleRecommend(params) {
  const t0 = performance.now();
  const st = loadBundle();
  const userId = parseInt(params.user_id ?? "5", 10);
  const k = Math.min(50, Math.max(1, parseInt(params.k ?? "10", 10)));
  const likedStr = String(params.liked ?? "").trim();
  const genresStr = String(params.genres ?? "").trim();
  const variantReq = String(params.variant ?? "").trim().toUpperCase();
  const bypassCache = String(params.bypass_cache ?? "false") === "true";

  const variant = variantReq === "A" || variantReq === "B" ? variantReq : variantFor(userId);
  const rankerModel = variant === "A" ? "ranker_wd" : "ranker";
  const cacheKey = `rec:v${st.cfg.model_version}:${variant}:${userId}:${k}:${likedStr}|${genresStr}`;

  if (!bypassCache && _cache.has(cacheKey)) {
    const hit = _cache.get(cacheKey);
    if (Date.now() < hit.exp) {
      _cacheHits++;
      return {
        user_id: userId,
        variant,
        ranker_model: rankerModel,
        source: "cache",
        items: hit.items,
        history_preview: hit.history_preview,
        latency_ms: Number((performance.now() - t0).toFixed(2)),
      };
    }
    _cache.delete(cacheKey);
  }
  _cacheMisses++;

  const likedIds = likedStr ? likedStr.split(",").map((x) => parseInt(x.trim(), 10)).filter((n) => !isNaN(n)) : [];
  const genreList = genresStr ? genresStr.split(",").map((g) => g.trim()).filter(Boolean) : [];

  let source = "popularity";
  let outItems = [];
  let historyPreview = [];

  const uProfile = st.users[String(userId)];
  if (uProfile && !likedIds.length && !genreList.length) {
    historyPreview = uProfile.history
      .slice(-8)
      .reverse()
      .map((id) => st.itemByIdx.get(id))
      .filter(Boolean);
    if (uProfile.history.length >= (st.cfg.min_history_for_warm || 5)) {
      source = "warm";
      outItems = warmRecommend(st, uProfile, variant, k);
    } else if (uProfile.history.length >= 1) {
      source = "warm_blend";
      const wItems = warmRecommend(st, uProfile, variant, Math.ceil(k / 2));
      const excl = new Set([...uProfile.history, ...wItems.map((x) => x.item_idx)]);
      const pItems = popularFallback(st, k - wItems.length, [], excl);
      outItems = [...wItems, ...pItems].slice(0, k);
    } else {
      outItems = popularFallback(st, k);
    }
  } else if (likedIds.length) {
    source = "content_based";
    historyPreview = likedIds.map((id) => st.itemByIdx.get(id)).filter(Boolean);
    outItems = contentBasedFromLiked(st, likedIds, k);
  } else if (genreList.length) {
    source = "popularity_genre";
    outItems = popularFallback(st, k, genreList);
  } else {
    source = "popularity";
    outItems = popularFallback(st, k);
  }

  if (_cache.size > 2000) _cache.clear();
  _cache.set(cacheKey, { exp: Date.now() + 3600_000, items: outItems, history_preview: historyPreview });

  return {
    user_id: userId,
    variant,
    ranker_model: rankerModel,
    source,
    items: outItems,
    history_preview: historyPreview,
    latency_ms: Number((performance.now() - t0).toFixed(2)),
  };
}

export function routeApiRequest(pathname, queryObj) {
  const st = loadBundle();
  const cleanPath = pathname.replace(/^\/\.netlify\/functions\/api/, "") || "/";

  if (cleanPath === "/health" || cleanPath === "/api/health") {
    const total = _cacheHits + _cacheMisses;
    return {
      status: "ok",
      backend: { active_engine: "serverless_exported_weights", triton_live: false },
      cache: {
        backend: "serverless_memory_ttl",
        model_version: String(st.cfg.model_version),
        hits: _cacheHits,
        misses: _cacheMisses,
        hit_rate: total > 0 ? Number((_cacheHits / total).toFixed(4)) : 0,
      },
      n_catalog_items: st.nTotalCatalog,
      n_warm_users: st.nTotalWarmUsers,
    };
  }

  if (cleanPath === "/recommend" || cleanPath === "/api/recommend") {
    return handleRecommend(queryObj);
  }

  if (cleanPath === "/api/compare") {
    const uid = queryObj.user_id ?? "5";
    const k = queryObj.k ?? "10";
    const resA = handleRecommend({ user_id: uid, k, variant: "A", bypass_cache: "true" });
    const resB = handleRecommend({ user_id: uid, k, variant: "B", bypass_cache: "true" });
    const setA = new Set(resA.items.map((x) => x.item_idx));
    const overlap = resB.items.filter((x) => setA.has(x.item_idx)).length;
    return {
      user_id: parseInt(uid, 10),
      source: resA.source,
      history_preview: resA.history_preview,
      overlap_at_k: overlap,
      variant_A: { model: "Wide & Deep (ranker_wd)", items: resA.items, latency_ms: resA.latency_ms },
      variant_B: { model: "DCN-V2 (ranker)", items: resB.items, latency_ms: resB.latency_ms },
    };
  }

  if (cleanPath === "/api/items") {
    const q = String(queryObj.q ?? "").trim().toLowerCase();
    const genre = String(queryObj.genre ?? "").trim().toLowerCase();
    const limit = Math.min(100, parseInt(queryObj.limit ?? "40", 10));
    let filtered = [...st.items].sort((a, b) => b.n_pos - a.n_pos);
    if (q) filtered = filtered.filter((it) => it.title.toLowerCase().includes(q));
    if (genre) filtered = filtered.filter((it) => it.genres.toLowerCase().includes(genre));
    return { genres: st.genres, items: filtered.slice(0, limit) };
  }

  if (cleanPath === "/api/users") {
    const profiles = Object.values(st.users).slice(0, 16).map((u) => {
      const gCounts = {};
      u.history.forEach((id) => {
        const it = st.itemByIdx.get(id);
        if (it) {
          it.genres.split("|").forEach((g) => {
            if (g && g !== "(no genres listed)") gCounts[g] = (gCounts[g] || 0) + 1;
          });
        }
      });
      const topGenres = Object.entries(gCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
        .map(([g]) => g);
      const recentTitles = u.history
        .slice(-5)
        .reverse()
        .map((id) => st.itemByIdx.get(id)?.title || "?");
      return {
        user_id: u.user_id,
        history_count: u.history.length,
        segment: u.history.length >= (st.cfg.min_history_for_warm || 5) ? "warm" : "lukewarm",
        assigned_variant: variantFor(u.user_id),
        top_genres: topGenres,
        recent_titles: recentTitles,
      };
    });
    return { users: profiles };
  }

  if (cleanPath === "/api/metrics") {
    const total = _cacheHits + _cacheMisses;
    return {
      metrics: st.telemetry.metrics,
      faiss_benchmark: st.telemetry.faiss_benchmark,
      split_summary: st.telemetry.split_summary,
      retrieval_train: st.telemetry.retrieval_train,
      ranking_train: st.telemetry.ranking_train,
      cache: {
        backend: "serverless_memory_ttl",
        model_version: String(st.cfg.model_version),
        hits: _cacheHits,
        misses: _cacheMisses,
        hit_rate: total > 0 ? Number((_cacheHits / total).toFixed(4)) : 0,
      },
      backend: { active_engine: "serverless_exported_weights", triton_live: false },
    };
  }

  if (cleanPath === "/api/ab-results") {
    return st.telemetry.ab_summary;
  }

  return { status: "ok", message: "Two-Stage RecSys Serverless API" };
}
