import unittest

from rocycle_robot.db_bridge_logic import build_processing_payload


class DatabaseBridgeLogicTests(unittest.TestCase):
    def test_completed_item_payload_and_stable_external_id(self):
        state = {
            "last": {
                "ts": 100.5,
                "item": "can",
                "dest": "can_bin",
                "confidence": 0.91,
                "net_weight_kg": 0.016,
            }
        }
        first = build_processing_payload(state)
        second = build_processing_payload(state)

        self.assertEqual(first, second)
        external_id, payload = first
        self.assertEqual(external_id, "ui-state:100.5:can")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["result"], "completed")
        self.assertEqual(payload["net_weight_kg"], 0.016)

    def test_review_item_gets_default_reason(self):
        built = build_processing_payload(
            {"last": {"ts": 101, "item": "plastic_bag", "dest": "review_bin"}}
        )
        _, payload = built
        self.assertEqual(payload["status"], "review_pending")
        self.assertIsNone(payload["result"])
        self.assertEqual(payload["review_reason"], "manual_review_required")

    def test_legacy_class_aliases_follow_current_model(self):
        _, labeled = build_processing_payload(
            {"last": {"ts": 102, "item": "pet_bottle_labeled", "dest": "human_handoff"}}
        )
        _, plastic = build_processing_payload(
            {"last": {"ts": 103, "item": "pet_bottle_unlabeled", "dest": "plastic_bin"}}
        )
        self.assertEqual(labeled["predicted_class"], "pet_labeled")
        self.assertEqual(plastic["predicted_class"], "plastic")

    def test_in_progress_and_unknown_classes_are_not_queued(self):
        self.assertIsNone(build_processing_payload({"last": {"item": "can"}}))
        with self.assertRaises(ValueError):
            build_processing_payload(
                {"last": {"ts": 104, "item": "unknown", "dest": "review_bin"}}
            )


if __name__ == "__main__":
    unittest.main()
