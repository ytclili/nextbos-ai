from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_model_params_validation_rejects_invalid_temperature():
    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        json={
            "thread_id": "thread-1",
            "user_id": "user-1",
            "message": "今天吃什么？",
            "model_params": {"temperature": 3},
        },
    )

    assert response.status_code == 422


def test_chat_response_schema_contains_frontend_render_items():
    schema = app.openapi()["components"]["schemas"]["ChatResponse"]

    assert "status" in schema["properties"]
    assert "thread_id" in schema["properties"]
    assert "items" in schema["properties"]
