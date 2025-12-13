from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

import src.api.main as api_main

def make_test_client():
    api_main.model = MagicMock()
    api_main.graph = MagicMock()

    # Fake model bundle for /health + startup
    api_main.model.bundle = {
        "metrics": {
            "accuracy": 0.8,
            "top3_accuracy": 0.9,
        },
        "metadata": {
            "version": "test",
            "top_k": 3,
        },
    }

    # Fake prediction
    api_main.model.predict_top_k.return_value = {
        "predictions": [
            {"node": "abc", "probability": 0.7},
            {"node": "def", "probability": 0.2},
            {"node": "ghi", "probability": 0.1},
        ],
        "confidence": 0.7,
        "entropy": 0.5,
    }

    # Fake graph
    api_main.graph.node_summary.return_value = {
        "node": "abc",
        "degree": 2,
        "total_transitions": 10,
        "outgoing_transitions": {"x": 5, "y": 5},
        "outgoing_probabilities": {"x": 0.5, "y": 0.5},
        "top_transitions": [
            {"node": "x", "probability": 0.5},
            {"node": "y", "probability": 0.5},
        ],
    }

    return TestClient(api_main.app)


def test_health_endpoint():
    client = make_test_client()

    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "ok"
    assert body["model_version"] == "test"
    assert body["top_k"] == 3


def test_predict_next_node():
    client = make_test_client()

    payload = {
        "prev_node": "a",
        "curr_node": "b",
        "hour": 12,
        "day_of_week": 2,
        "call_type": "A",
    }

    resp = client.post("/predict_next_node", json=payload)

    assert resp.status_code == 200

    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 3
    assert "node" in data[0]
    assert "probability" in data[0]


def test_graph_node_info(): 
    client = make_test_client()

    resp = client.get("/graph_node_info/abc")

    assert resp.status_code == 200
    body = resp.json()

    assert body["node"] == "abc"
    assert body["degree"] == 2


def test_graph_node_info_not_found():
    client = make_test_client()
    api_main.graph.node_summary.return_value = {
        "node": "zzz",
        "degree": 0,
        "total_transitions": 0,
        "outgoing_transitions": {},
        "outgoing_probabilities": {},
        "top_transitions": [],
    }

    resp = client.get("/graph_node_info/zzz")
    assert resp.status_code == 404
