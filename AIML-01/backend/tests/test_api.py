from fastapi.testclient import TestClient
# Requires running infra + champion; skipped in CI without services.
import pytest
pytestmark = pytest.mark.skip(reason="needs Redis/Postgres/MLflow + champion alias")
def test_health(client): assert client.get("/health").status_code == 200
def test_score_unknown_transaction_does_not_crash(client):
    r = client.post("/score", json={"transaction_id": -1, "uid": "nope"}); assert r.status_code in (200, 404, 422)
