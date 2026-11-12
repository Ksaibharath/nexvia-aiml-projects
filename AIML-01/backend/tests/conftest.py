import pytest
@pytest.fixture
def client():
    from fraud.serving.api import app
    with TestClient(app) as c: yield c
