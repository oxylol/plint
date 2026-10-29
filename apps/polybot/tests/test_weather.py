import unittest
from datetime import date

from polybot.weather import Bucket, parse_question, parse_station, pick_bucket, target_value, Forecast


class ParseTests(unittest.TestCase):
    def test_question_forms(self):
        b = parse_question("Will the highest temperature in Paris be 24°C on October 30?", 2026)
        self.assertEqual((b.kind, b.city, b.unit, b.lo, b.hi, b.day, b.label), ("highest", "Paris", "C", 24.0, 24.0, date(2026, 10, 30), "24°C"))
        b = parse_question("Will the highest temperature in Seoul (Incheon) be 19°C or higher on October 30?", 2026)
        self.assertEqual((b.city, b.lo, b.hi, b.label), ("Seoul (Incheon)", 19.0, None, "19°C or higher"))
        b = parse_question("Will the lowest temperature in Munich be -3°C or below on October 30?", 2026)
        self.assertEqual((b.kind, b.lo, b.hi), ("lowest", None, -3.0))
        b = parse_question("Will the highest temperature in Dallas be between 66-67°F on October 30?", 2026)
        self.assertEqual((b.unit, b.lo, b.hi, b.label), ("F", 66.0, 67.0, "66–67°F"))
        self.assertIsNone(parse_question("Will Bitcoin reach $82,500 in October?", 2026))

    def test_station(self):
        self.assertEqual(parse_station("…reported at Paris-Le Bourget Airport (LFPB) on 30 Oct '26.", "Paris"), "LFPB")
        self.assertEqual(parse_station("no code here", "Hong Kong"), "VHHH")
        self.assertIsNone(parse_station("no code here", "Atlantis"))

    def test_pick_bucket_and_target(self):
        mk = lambda lo, hi: Bucket("highest", "Paris", "C", lo, hi, date(2026, 10, 30), "")  # noqa: E731
        buckets = [mk(None, 9), mk(10, 10), mk(11, 11), mk(12, 12), mk(13, None)]
        self.assertEqual(pick_bucket(buckets, 11.4).label, "11°C")
        self.assertEqual(pick_bucket(buckets, 11.5).label, "12°C")
        self.assertEqual(pick_bucket(buckets, 14.2).label, "13°C or higher")
        self.assertEqual(pick_bucket(buckets, 3.0).label, "9°C or below")
        self.assertAlmostEqual(target_value(25.8, 2, "C"), 27.8)
        self.assertAlmostEqual(target_value(20.0, 2, "F"), 71.6)
        f = Forecast(day=date(2026, 10, 30), highs={"a": 24.4, "b": 25.7, "c": 25.0})
        self.assertEqual(f.agg("highest", "max"), 25.7)
        self.assertEqual(f.agg("highest", "median"), 25.0)
        self.assertAlmostEqual(f.spread("highest"), 1.3)


if __name__ == "__main__":
    unittest.main()
