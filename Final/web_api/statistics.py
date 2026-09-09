"""Pure helpers for day/week/month statistics."""

from __future__ import annotations

from datetime import datetime, timedelta


CLASS_NAMES = (
    "battery",
    "can",
    "paper",
    "pet_labeled",
    "plastic",
    "plastic_bag",
)


def period_definition(
    period: str, now: datetime
) -> tuple[datetime, str, str, list[str]]:
    if period == "day":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return (
            start,
            f"오늘 · {now:%Y.%m.%d}",
            "시간대별 처리량",
            ["00–03", "04–07", "08–11", "12–15", "16–19", "20–23"],
        )
    if period == "week":
        start = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end = start + timedelta(days=6)
        return (
            start,
            f"이번 주 · {start:%m.%d}–{end:%m.%d}",
            "요일별 처리량",
            ["월", "화", "수", "목", "금", "토", "일"],
        )
    if period == "month":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return (
            start,
            f"이번 달 · {now:%Y년 %m월}",
            "주차별 처리량",
            ["1주", "2주", "3주", "4주", "5주"],
        )
    raise ValueError("period는 day, week, month 중 하나여야 합니다.")


def period_bucket_index(period: str, value: datetime) -> int:
    if period == "day":
        return value.hour // 4
    if period == "week":
        return value.weekday()
    if period == "month":
        return (value.day - 1) // 7
    raise ValueError("period는 day, week, month 중 하나여야 합니다.")
