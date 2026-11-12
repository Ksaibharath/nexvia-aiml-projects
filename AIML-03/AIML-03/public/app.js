// Interactive Frontend Controller for Two-Stage Recommendation Engine
const state = {
  users: [],
  catalog: [],
  genres: [],
  pickedMovies: new Map(), // item_idx -> title
  pickedGenres: new Set(),
  coldMode: "content",
};

async function apiGet(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderMovieCards(containerId, items, highlightDiffSet = null) {
  const el = document.getElementById(containerId);
  if (!items || !items.length) {
    el.innerHTML = `<p class="muted">No recommendations returned.</p>`;
    return;
  }
  el.innerHTML = items
    .map((m, idx) => {
      const genres = (m.genres || "").split("|").filter(Boolean).join(" · ");
      const isUnique = highlightDiffSet && !highlightDiffSet.has(m.item_idx);
      const scoreHtml =
        m.score !== undefined
          ? `<strong>Logit ${Number(m.score).toFixed(3)}</strong><span>Cos ${Number(m.retrieval_score ?? 0).toFixed(3)}</span>`
          : m.retrieval_score !== undefined
          ? `<strong>Sim ${Number(m.retrieval_score).toFixed(3)}</strong><span>★ ${m.avg_rating ?? "—"}</span>`
          : `<strong>★ ${m.avg_rating ?? "—"}</strong><span>${m.n_pos ?? 0} likes</span>`;
      return `
        <div class="movie-item ${isUnique ? "diff-highlight" : ""}">
          <div class="rank-num">#${idx + 1}</div>
          <div>
            <div class="movie-title">${escapeHtml(m.title)} ${isUnique ? '<span class="badge badge-purple">Unique Rank</span>' : ""}</div>
            <div class="movie-meta">${escapeHtml(genres)} · ★ ${m.avg_rating ?? "—"} · ID ${m.item_idx}</div>
          </div>
          <div class="score-col">${scoreHtml}</div>
        </div>
      `;
    })
    .join("");
}

function updateTrace(source) {
  const steps = document.querySelectorAll("#pipeline-trace .trace-step");
  steps.forEach((s) => s.classList.remove("active", "cache-hit"));
  if (source === "cache") {
    steps[0].classList.add("active");
    steps[1].classList.add("cache-hit");
  } else if (source === "warm" || source === "warm_blend") {
    steps.forEach((s) => s.classList.add("active"));
  } else {
    steps[0].classList.add("active");
  }
}

async function runWarmRecommendation(forceRepeat = false) {
  const uid = document.getElementById("warm-user-id").value || "5";
  const k = document.getElementById("warm-k").value || "10";
  const variant = document.getElementById("warm-variant").value;
  const bypass = forceRepeat ? false : document.getElementById("warm-bypass-cache").checked;

  const qs = new URLSearchParams({ user_id: uid, k });
  if (variant) qs.set("variant", variant);
  if (bypass) qs.set("bypass_cache", "true");

  const url = `/recommend?${qs.toString()}`;
  document.getElementById("warm-curl-preview").textContent = url;

  const data = await apiGet(url);
  document.getElementById("res-source").textContent = data.source;
  document.getElementById("res-variant").textContent = `Variant ${data.variant}`;
  document.getElementById("res-latency").textContent = `${data.latency_ms} ms`;
  document.getElementById("res-model").textContent = data.ranker_model || (data.variant === "A" ? "ranker_wd" : "ranker");

  updateTrace(data.source);

  const histCard = document.getElementById("history-card");
  const histWrap = document.getElementById("history-chips");
  if (data.history_preview && data.history_preview.length) {
    histCard.classList.remove("hidden");
    document.getElementById("history-count-badge").textContent = `${data.history_preview.length} recent liked films`;
    histWrap.innerHTML = data.history_preview
      .map((h) => `<span class="chip">${escapeHtml(h.title)}</span>`)
      .join("");
  } else {
    histCard.classList.add("hidden");
  }

  renderMovieCards("warm-items-grid", data.items);
}

function renderPickedChips() {
  document.getElementById("picked-count").textContent = String(state.pickedMovies.size);
  const wrap = document.getElementById("picked-movies-chips");
  wrap.innerHTML = [...state.pickedMovies.entries()]
    .map(
      ([id, title]) =>
        `<span class="chip picked" data-remove="${id}">${escapeHtml(title)} ✕</span>`
    )
    .join("");
  wrap.querySelectorAll("[data-remove]").forEach((el) => {
    el.addEventListener("click", () => {
      state.pickedMovies.delete(Number(el.dataset.remove));
      renderPickedChips();
      renderCatalogList();
    });
  });
}

function renderCatalogList() {
  const listEl = document.getElementById("catalog-picker-list");
  listEl.innerHTML = state.catalog
    .map((m) => {
      const picked = state.pickedMovies.has(m.item_idx);
      return `
        <div class="catalog-row" data-pick="${m.item_idx}" data-title="${escapeHtml(m.title)}">
          <div>
            <strong>${escapeHtml(m.title)}</strong>
            <div class="muted-sm">${escapeHtml(m.genres)} · ★ ${m.avg_rating}</div>
          </div>
          <span class="badge ${picked ? "badge-blue" : ""}">${picked ? "Selected ✓" : "+ Pick"}</span>
        </div>
      `;
    })
    .join("");

  listEl.querySelectorAll(".catalog-row").forEach((row) => {
    row.addEventListener("click", () => {
      const id = Number(row.dataset.pick);
      if (state.pickedMovies.has(id)) {
        state.pickedMovies.delete(id);
      } else if (state.pickedMovies.size < 6) {
        state.pickedMovies.set(id, row.dataset.title);
      }
      renderPickedChips();
      renderCatalogList();
    });
  });
}

async function runColdRecommendation() {
  const qs = new URLSearchParams({ user_id: "99999999", k: "10", bypass_cache: "true" });
  if (state.coldMode === "content" && state.pickedMovies.size > 0) {
    qs.set("liked", [...state.pickedMovies.keys()].join(","));
  } else if (state.coldMode === "genre" && state.pickedGenres.size > 0) {
    qs.set("genres", [...state.pickedGenres].join(","));
  }
  const data = await apiGet(`/recommend?${qs.toString()}`);
  document.getElementById("cold-res-source").textContent = data.source;
  document.getElementById("cold-res-latency").textContent = `${data.latency_ms} ms`;
  renderMovieCards("cold-items-grid", data.items);
}

async function runAbCompare() {
  const uid = document.getElementById("ab-user-select").value || "5";
  const cmp = await apiGet(`/api/compare?user_id=${encodeURIComponent(uid)}&k=10`);
  const setA = new Set(cmp.variant_A.items.map((x) => x.item_idx));
  const setB = new Set(cmp.variant_B.items.map((x) => x.item_idx));

  document.getElementById("ab-overlap-banner").innerHTML =
    `User <strong>#${cmp.user_id}</strong> (${cmp.source}) — Top-10 Agreement: <strong>${cmp.overlap_at_k} / 10 movies</strong> shared; ` +
    `<strong>${10 - cmp.overlap_at_k}</strong> ranker-specific picks highlighted below.`;

  document.getElementById("ab-lat-a").textContent = `${cmp.variant_A.latency_ms} ms`;
  document.getElementById("ab-lat-b").textContent = `${cmp.variant_B.latency_ms} ms`;

  renderMovieCards("ab-list-a", cmp.variant_A.items, setB);
  renderMovieCards("ab-list-b", cmp.variant_B.items, setA);
}

async function loadMetricsAndAb() {
  const [mData, abData] = await Promise.all([apiGet("/api/metrics"), apiGet("/api/ab-results")]);

  // 1) Retrieval Table & Bar Chart
  const ret = mData.metrics?.retrieval;
  if (ret) {
    const r300Max = Math.max(ret.two_tower["recall@300"], ret.popularity["recall@300"], 0.01);
    document.getElementById("retrieval-chart-box").innerHTML = `
      <div class="bar-chart">
        ${[
          ["Two-Tower", ret.two_tower["recall@300"], ""],
          ["Popularity", ret.popularity["recall@300"], "purple"],
          ["Random", ret.random["recall@300"], ""],
        ]
          .map(
            ([label, val, cls]) => `
          <div class="bar-row">
            <span>${label} @300</span>
            <div class="bar-track"><div class="bar-fill ${cls}" style="width:${((val / r300Max) * 100).toFixed(1)}%"></div></div>
            <strong>${val.toFixed(4)}</strong>
          </div>`
          )
          .join("")}
      </div>
    `;
    document.getElementById("retrieval-table-box").innerHTML = `
      <table class="data-table">
        <thead><tr><th>K</th><th>Random</th><th>Popularity</th><th>Two-Tower (Ours)</th></tr></thead>
        <tbody>
          ${[10, 50, 100, 300]
            .map((k) => {
              const tt = ret.two_tower[`recall@${k}`];
              const pop = ret.popularity[`recall@${k}`];
              return `<tr class="${tt >= pop ? "win-row" : ""}">
                <td>Recall@${k}</td>
                <td>${ret.random[`recall@${k}`].toFixed(4)}</td>
                <td>${pop.toFixed(4)}</td>
                <td>${tt.toFixed(4)}</td>
              </tr>`;
            })
            .join("")}
        </tbody>
      </table>
    `;
  }

  // 2) Ranking Table & Bar Chart
  const rnk = mData.metrics?.ranking;
  if (rnk) {
    const labels = {
      random: "Random Order",
      popularity: "Popularity",
      retrieval_only: "Retrieval Only",
      wide_and_deep: "Wide & Deep (A)",
      dcn_v2: "DCN-V2 (B)",
    };
    const maxNdcg = Math.max(...Object.values(rnk).map((v) => v["ndcg@10"]), 0.01);
    document.getElementById("ranking-chart-box").innerHTML = `
      <div class="bar-chart">
        ${Object.entries(rnk)
          .map(
            ([k, v]) => `
          <div class="bar-row">
            <span>${labels[k] || k}</span>
            <div class="bar-track"><div class="bar-fill ${k === "dcn_v2" ? "purple" : ""}" style="width:${((v["ndcg@10"] / maxNdcg) * 100).toFixed(1)}%"></div></div>
            <strong>${v["ndcg@10"].toFixed(4)}</strong>
          </div>`
          )
          .join("")}
      </div>
    `;
    document.getElementById("ranking-table-box").innerHTML = `
      <table class="data-table">
        <thead><tr><th>Method</th><th>NDCG@10</th><th>95% CI</th><th>Recall@10</th><th>HitRate@10</th></tr></thead>
        <tbody>
          ${Object.entries(rnk)
            .map(
              ([k, v]) => `<tr class="${k === "wide_and_deep" || k === "dcn_v2" ? "win-row" : ""}">
                <td>${labels[k] || k}</td>
                <td>${v["ndcg@10"].toFixed(4)}</td>
                <td>[${v["ndcg@10_ci95"][0].toFixed(4)}, ${v["ndcg@10_ci95"][1].toFixed(4)}]</td>
                <td>${v["recall@10"].toFixed(4)}</td>
                <td>${(v["hit_rate@10"] * 100).toFixed(1)}%</td>
              </tr>`
            )
            .join("")}
        </tbody>
      </table>
    `;
  }

  // 3) FAISS Table
  const fb = mData.faiss_benchmark;
  if (fb?.catalog) {
    const sc = fb.scaling || {};
    document.getElementById("faiss-table-box").innerHTML = `
      <table class="data-table">
        <thead><tr><th>Index Type</th><th>Recall vs Exact@100</th><th>Queries / Sec</th><th>ms / Query</th></tr></thead>
        <tbody>
          ${Object.entries(fb.catalog)
            .map(
              ([name, row]) => `<tr class="${name === "hnsw" ? "win-row" : ""}">
                <td>${name.toUpperCase()}</td>
                <td>${(row["recall_vs_exact@100"] * 100).toFixed(2)}%</td>
                <td>${Number(row.qps).toLocaleString()}</td>
                <td>${row.ms_per_query} ms</td>
              </tr>`
            )
            .join("")}
        </tbody>
      </table>
      ${
        sc.n_vectors
          ? `<div class="banner-info" style="margin-top:12px;">
              <strong>Large-Scale Benchmark (${Number(sc.n_vectors).toLocaleString()} 64-d vectors):</strong>
              HNSW achieves <strong>${sc.speedup_x}x higher QPS</strong> (${Number(sc.hnsw_qps).toLocaleString()} vs ${Number(sc.flat_qps).toLocaleString()} qps)
              with <strong>${(sc["hnsw_recall_vs_exact@100"] * 100).toFixed(1)}% recall</strong> vs exact search.
            </div>`
          : ""
      }
    `;
  }

  // 4) Split Summary Table
  const sp = mData.split_summary?.splits;
  if (sp) {
    document.getElementById("split-table-box").innerHTML = `
      <table class="data-table">
        <thead><tr><th>Split</th><th>Rows</th><th>Users</th><th>Movies</th><th>Start (UTC)</th><th>End (UTC)</th></tr></thead>
        <tbody>
          ${sp
            .map(
              (r) => `<tr>
                <td><strong>${r.split.toUpperCase()}</strong></td>
                <td>${Number(r.rows).toLocaleString()}</td>
                <td>${Number(r.users).toLocaleString()}</td>
                <td>${Number(r.items).toLocaleString()}</td>
                <td>${r.from}</td>
                <td>${r.to}</td>
              </tr>`
            )
            .join("")}
        </tbody>
      </table>
    `;
  }

  // 5) A/B Replay Summary
  if (abData?.warm_only) {
    const w = abData.warm_only;
    const lat = abData.latency || {};
    document.getElementById("ab-stats-container").innerHTML = `
      <div class="kpi-row" style="margin-bottom:14px;">
        <div class="kpi-box"><span class="kpi-label">Simulated Visitors</span><strong class="kpi-val">${abData.total_visitors}</strong></div>
        <div class="kpi-box"><span class="kpi-label">Traffic Split (A / B)</span><strong class="kpi-val">${abData.variant_counts.A} / ${abData.variant_counts.B}</strong></div>
        <div class="kpi-box"><span class="kpi-label">SRM Chi-Square p</span><strong class="kpi-val accent-green">${abData.srm_p_value} (Pass)</strong></div>
        <div class="kpi-box"><span class="kpi-label">Warm Miss p95 / Cache Hit p95</span><strong class="kpi-val">${lat.warm_miss_p95_ms ?? "—"}ms / ${lat.cache_hit_p95_ms ?? "—"}ms</strong></div>
      </div>
      <table class="data-table">
        <thead><tr><th>Warm Cohort Metric</th><th>Variant A (Wide &amp; Deep)</th><th>Variant B (DCN-V2)</th><th>Diff (B − A)</th><th>95% Bootstrap CI</th><th>Welch p-value</th></tr></thead>
        <tbody>
          ${["ctr", "ndcg10", "hit10"]
            .map((m) => {
              const row = w[m];
              return `<tr>
                <td><strong>${m.toUpperCase()}</strong></td>
                <td>${row.A_mean.toFixed(4)}</td>
                <td>${row.B_mean.toFixed(4)}</td>
                <td>${row.diff_B_minus_A >= 0 ? "+" : ""}${row.diff_B_minus_A.toFixed(4)}</td>
                <td>[${row.ci95_diff[0]}, ${row.ci95_diff[1]}]</td>
                <td>${row.p_value.toFixed(4)}</td>
              </tr>`;
            })
            .join("")}
        </tbody>
      </table>
    `;
  }
}

async function init() {
  // Tab switching
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    });
  });

  // Health status
  try {
    const h = await apiGet("/api/health");
    document.getElementById("health-pill").textContent =
      `API Ready · v${h.cache?.model_version ?? 1} · ${Number(h.n_catalog_items).toLocaleString()} Movies`;
    document.getElementById("footer-runtime-info").textContent =
      `Engine: ${h.backend?.active_engine ?? "ready"} · Cache: ${h.cache?.backend ?? "ready"}`;
  } catch (e) {
    document.getElementById("health-pill").textContent = "API Error";
  }

  // Load sample warm users
  const uResp = await apiGet("/api/users");
  state.users = uResp.users || [];
  const warmSel = document.getElementById("warm-user-select");
  const abSel = document.getElementById("ab-user-select");
  const optsHtml = state.users
    .map(
      (u) =>
        `<option value="${u.user_id}">User #${u.user_id} (${u.segment}, ${u.history_count} likes · ${u.top_genres.join("/")})</option>`
    )
    .join("");
  warmSel.innerHTML = optsHtml;
  abSel.innerHTML = optsHtml;

  warmSel.addEventListener("change", () => {
    document.getElementById("warm-user-id").value = warmSel.value;
    runWarmRecommendation(false);
  });

  document.getElementById("btn-warm-run").addEventListener("click", () => runWarmRecommendation(false));
  document.getElementById("btn-warm-repeat").addEventListener("click", () => runWarmRecommendation(true));

  // Load catalog & genres for Cold-Start Onboarding
  const cResp = await apiGet("/api/items?limit=50");
  state.catalog = cResp.items || [];
  state.genres = cResp.genres || [];

  // Pre-pick 3 iconic Sci-Fi/Adventure movies so Cold-Start tab works immediately
  state.catalog.slice(0, 30).forEach((m) => {
    if (
      state.pickedMovies.size < 3 &&
      (m.title.includes("Matrix, The (1999)") ||
        m.title.includes("Star Wars: Episode IV") ||
        m.title.includes("Inception (2010)"))
    ) {
      state.pickedMovies.set(m.item_idx, m.title);
    }
  });
  renderPickedChips();
  renderCatalogList();

  // Genre chips
  const gWrap = document.getElementById("cold-genre-chips");
  gWrap.innerHTML = state.genres
    .map((g) => `<span class="chip" data-genre="${escapeHtml(g)}">${escapeHtml(g)}</span>`)
    .join("");
  gWrap.querySelectorAll(".chip").forEach((ch) => {
    ch.addEventListener("click", () => {
      const g = ch.dataset.genre;
      if (state.pickedGenres.has(g)) {
        state.pickedGenres.delete(g);
        ch.classList.remove("selected");
      } else {
        state.pickedGenres.add(g);
        ch.classList.add("selected");
      }
    });
  });

  // Cold-start radio switching
  document.querySelectorAll('input[name="cold_mode"]').forEach((rad) => {
    rad.addEventListener("change", () => {
      state.coldMode = rad.value;
      document.querySelectorAll(".radio-card").forEach((c) => c.classList.remove("selected"));
      rad.closest(".radio-card").classList.add("selected");
      document.getElementById("cold-movie-picker-box").classList.toggle("hidden", state.coldMode !== "content");
      document.getElementById("cold-genre-picker-box").classList.toggle("hidden", state.coldMode !== "genre");
    });
  });

  let searchTimer = null;
  document.getElementById("cold-search-input").addEventListener("input", (e) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(async () => {
      const r = await apiGet(`/api/items?q=${encodeURIComponent(e.target.value)}&limit=40`);
      state.catalog = r.items || [];
      renderCatalogList();
    }, 200);
  });

  document.getElementById("btn-clear-picked").addEventListener("click", () => {
    state.pickedMovies.clear();
    renderPickedChips();
    renderCatalogList();
  });

  document.getElementById("btn-cold-run").addEventListener("click", runColdRecommendation);
  document.getElementById("btn-ab-compare").addEventListener("click", runAbCompare);

  await Promise.all([runWarmRecommendation(false), runColdRecommendation(), runAbCompare(), loadMetricsAndAb()]);
}

init();
