import unittest
from datetime import datetime

from web_api.statistics import period_bucket_index, period_definition


class StatisticsPeriodTests(unittest.TestCase):
    def test_day_definition_and_buckets(self):
        now = datetime(2026, 9, 9, 16, 20)
        start, _, title, labels = period_definition("day", now)

        self.assertEqual(start, datetime(2026, 9, 9))
        self.assertEqual(title, "시간대별 처리량")
        self.assertEqual(len(labels), 6)
        self.assertEqual(period_bucket_index("day", datetime(2026, 9, 9, 0, 1)), 0)
        self.assertEqual(period_bucket_index("day", datetime(2026, 9, 9, 23, 59)), 5)

    def test_week_and_month_buckets(self):
        now = datetime(2026, 9, 9, 16, 20)
        week_start, _, _, week_labels = period_definition("week", now)
        month_start, _, _, month_labels = period_definition("month", now)

        self.assertEqual(week_start, datetime(2026, 9, 7))
        self.assertEqual(month_start, datetime(2026, 9, 1))
        self.assertEqual(len(week_labels), 7)
        self.assertEqual(len(month_labels), 5)
        self.assertEqual(period_bucket_index("week", datetime(2026, 9, 13)), 6)
        self.assertEqual(period_bucket_index("month", datetime(2026, 9, 30)), 4)


if __name__ == "__main__":
    unittest.main()
