"""Pure conversion logic shared by the ROS DB bridge and local tests."""

from __future__ import annotations

from typing import Any


CLASS_ALIASES = {
    "pet_bottle_labeled": "pet_labeled",
    "pet_bottle_unlabeled": "plastic",
    "pet_unlabeled": "plastic",
}

VALID_CLASSES = {
    "battery",
    "can",
    "paper",
    "pet_labeled",
    "plastic",
    "plastic_bag",
}


def build_processing_payload(
    state: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    """Convert one completed ``/ui/state`` cycle to the web API payload.

    A state without a completed destination is still in progress and returns
    ``None``. Unsupported class names are rejected instead of being retried
    forever by the HTTP queue.
    """

    last = state.get("last")
    if not isinstance(last, dict) or not last.get("dest") or not last.get("ts"):
        return None

    raw_item = last.get("item")
    if not raw_item:
        return None
    item = CLASS_ALIASES.get(str(raw_item), str(raw_item))
    if item not in VALID_CLASSES:
        raise ValueError(f"unsupported class: {raw_item}")

    destination = str(last["dest"])
    review_pending = destination == "review_bin"
    external_id = f"ui-state:{last['ts']}:{item}"
    payload = {
        "external_id": external_id,
        "predicted_class": item,
        "confidence": last.get("confidence"),
        "action": "robot_sort",
        "destination": destination,
        "result": None if review_pending else "completed",
        "status": "review_pending" if review_pending else "completed",
        "review_reason": (
            (last.get("reason") or "manual_review_required")
            if review_pending
            else last.get("reason")
        ),
        "net_weight_kg": last.get("net_weight_kg", last.get("weight")),
    }
    return external_id, payload
