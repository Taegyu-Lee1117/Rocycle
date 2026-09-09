from fastapi.testclient import TestClient

import web_api.main as api_module


class FakeDatabase:
    def initialize_schema(self):
        return None

    def ping(self):
        return True

    def dashboard(self):
        return {
            "counts": {"plastic_bin": 2, "review_bin": 1},
            "total": 2,
            "last": {"item": "plastic", "dest": "plastic_bin"},
            "review_pending": 1,
        }

    def list_reviews(self):
        return [
            {
                "id": 7,
                "predicted_class": "plastic",
                "confidence": 0.72,
                "review_reason": "weight_exceeded",
                "destination": "review_bin",
                "created_at": "2026-09-09T10:00:00",
            }
        ]

    def complete_review(self, processing_log_id, final_class, reviewed_by):
        return {
            "id": processing_log_id,
            "status": "completed",
            "final_class": final_class,
            "final_destination": "plastic_bin",
        }

    def create_processing_log(self, payload):
        return {"id": 8, "status": payload["status"], "destination": payload["destination"]}

    def statistics(self, period):
        return {
            "period": period,
            "range": "오늘",
            "completed": 2,
            "pending": 1,
            "failed": 0,
            "manual": 1,
            "timelineTitle": "시간대별 처리량",
            "labels": ["00–03"],
            "values": [2],
            "classes": {"plastic": 2},
        }


def test_ui_and_database_api(monkeypatch):
    monkeypatch.setattr(api_module, "database", FakeDatabase())
    with TestClient(api_module.app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/statistics.html").status_code == 200
        assert client.get("/api/health").json()["database"] == "ok"
        assert client.get("/api/dashboard").json()["review_pending"] == 1
        assert client.get("/api/reviews").json()["items"][0]["id"] == 7
        response = client.post(
            "/api/reviews/7/complete",
            json={"final_class": "plastic", "reviewed_by": "operator"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        assert client.get("/api/statistics?period=week").json()["period"] == "week"
