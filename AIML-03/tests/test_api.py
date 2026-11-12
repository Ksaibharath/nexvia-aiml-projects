"""Integration tests for FastAPI recommendation gateway, caching, cold-start, and A/B routing."""
from fastapi.testclient import TestClient

from src.serving.api import app, cache

client = TestClient(app)


def test_health():
    r = client.get("/health").json()
    assert r["status"] == "ok"


def test_unknown_user_falls_back():
    cache.clear_local()
    r = client.get("/recommend", params={"user_id": 99999999, "bypass_cache": True}).json()
    assert r["source"] == "popularity"
    assert len(r["items"]) == 10


def test_content_based_new_user():
    r = client.get("/recommend", params={"user_id": 99999998, "liked": "284,263,2016", "k": 10, "bypass_cache": True}).json()
    assert r["source"] == "content_based"
    assert len(r["items"]) == 10
    returned_ids = {x["item_idx"] for x in r["items"]}
    assert returned_ids.isdisjoint({284, 263, 2016})


def test_warm_user_and_cache_hit():
    cache.clear_local()
    users = client.get("/api/users").json()["users"]
    warm_uid = next(u["user_id"] for u in users if u["segment"] == "warm")

    first = client.get("/recommend", params={"user_id": warm_uid, "k": 10}).json()
    assert first["source"] == "warm"
    assert len(first["items"]) == 10

    second = client.get("/recommend", params={"user_id": warm_uid, "k": 10}).json()
    assert second["source"] == "cache"
    assert [x["item_idx"] for x in first["items"]] == [x["item_idx"] for x in second["items"]]


def test_ab_compare_endpoint():
    users = client.get("/api/users").json()["users"]
    warm_uid = next(u["user_id"] for u in users if u["segment"] == "warm")
    cmp_res = client.get("/api/compare", params={"user_id": warm_uid, "k": 10}).json()
    assert len(cmp_res["variant_A"]["items"]) == 10
    assert len(cmp_res["variant_B"]["items"]) == 10
