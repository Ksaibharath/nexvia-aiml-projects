"""Streamlit interactive demo UI for the Two-Stage Recommendation Engine (Part 11.6)."""
import os
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8080")

st.set_page_config(page_title="Two-Stage Movie RecSys", page_icon="🎬", layout="wide")
st.title("🎬 Large-Scale Two-Stage Personalized Recommendation Engine")
st.caption("MovieLens 25M · Two-Tower Retrieval · FAISS HNSW · DCN-V2 vs Wide & Deep · Redis Cache · A/B Router")

mode = st.sidebar.radio(
    "User Scenario",
    ["Warm User (Two-Stage Pipeline)", "New User — Cold-Start Onboarding", "A/B Ranker Comparison"],
)
k = st.sidebar.slider("Top-K Recommendations", min_value=5, max_value=25, value=10)
bypass = st.sidebar.checkbox("Bypass Redis Cache", value=False)

if mode == "Warm User (Two-Stage Pipeline)":
    users_resp = requests.get(f"{API_URL}/api/users", timeout=5).json().get("users", [])
    options = {f"User {u['user_id']} ({u['segment']}, {u['history_count']} likes, {', '.join(u['top_genres'])})": u["user_id"] for u in users_resp}
    choice = st.selectbox("Pick a Real MovieLens User Profile", list(options.keys())) if options else None
    custom_uid = st.number_input("Or enter any User ID", min_value=1, value=options[choice] if choice else 5, step=1)
    variant_override = st.selectbox("A/B Variant Override", ["Auto (MD5 Hash)", "A (Wide & Deep)", "B (DCN-V2)"])
    v_param = "" if variant_override.startswith("Auto") else variant_override[0]

    if st.button("Get Personalized Recommendations", type="primary"):
        r = requests.get(
            f"{API_URL}/recommend",
            params={"user_id": int(custom_uid), "k": k, "variant": v_param, "bypass_cache": bypass},
            timeout=10,
        ).json()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Source Path", r["source"])
        c2.metric("A/B Variant", f"{r['variant']} ({r.get('ranker_model', '')})")
        c3.metric("Latency", f"{r['latency_ms']} ms")
        c4.metric("Returned Items", len(r["items"]))

        if r.get("history_preview"):
            st.subheader("Recent Liked History")
            st.dataframe(pd.DataFrame(r["history_preview"])[["item_idx", "title", "genres", "year"]], use_container_width=True)

        st.subheader("Top-K Ranked Recommendations")
        st.dataframe(pd.DataFrame(r["items"]), use_container_width=True)

elif mode == "New User — Cold-Start Onboarding":
    catalog = requests.get(f"{API_URL}/api/items", params={"limit": 80}, timeout=5).json()
    movie_map = {f"{m['title']} [{m['genres']}]": m["item_idx"] for m in catalog.get("items", [])}
    picked_movies = st.multiselect("Pick 1–5 movies you like (leave empty for Popularity fallback)", list(movie_map.keys()))
    picked_genres = st.multiselect("Or filter by favourite genres (when no movies picked)", catalog.get("genres", []))

    if st.button("Recommend for New User (ID 99999999)", type="primary"):
        liked_str = ",".join(str(movie_map[m]) for m in picked_movies)
        genre_str = ",".join(picked_genres)
        r = requests.get(
            f"{API_URL}/recommend",
            params={"user_id": 99999999, "k": k, "liked": liked_str, "genres": genre_str, "bypass_cache": bypass},
            timeout=10,
        ).json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Cold-Start Strategy", r["source"])
        c2.metric("Latency", f"{r['latency_ms']} ms")
        c3.metric("Variant", r["variant"])
        st.dataframe(pd.DataFrame(r["items"]), use_container_width=True)

else:
    uid = st.number_input("Warm User ID to Compare Variant A vs Variant B", min_value=1, value=5, step=1)
    if st.button("Compare Wide & Deep (A) vs DCN-V2 (B)", type="primary"):
        cmp_res = requests.get(f"{API_URL}/api/compare", params={"user_id": int(uid), "k": k}, timeout=10).json()
        st.info(f"Top-{k} Overlap between Variant A and Variant B: **{cmp_res['overlap_at_k']} / {k}** items")
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Variant A — Wide & Deep")
            st.dataframe(pd.DataFrame(cmp_res["variant_A"]["items"]), use_container_width=True)
        with col_b:
            st.subheader("Variant B — DCN-V2")
            st.dataframe(pd.DataFrame(cmp_res["variant_B"]["items"]), use_container_width=True)
