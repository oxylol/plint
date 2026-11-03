import unittest
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from polybot.collector import day_summary, lock_state, payers, resolution_source, winning_bucket
from polybot.metar import Report, is_special, parse_temperature, whole_degrees
from polybot.weather import Bucket

PARIS = ZoneInfo("Europe/Paris")


def rep(hhmm_utc: str, tenths: int, day="2026-11-04", special=False):
    h, m = hhmm_utc.split(":")
    at = datetime.fromisoformat(f"{day}T{h}:{m}:00+00:00")
    return Report(at, tenths, f"{'SPECI' if special else 'METAR'} LFPB {at:%d%H%M}Z 18/12", special)


class MetarTests(unittest.TestCase):
    def test_rule_examples(self):
        self.assertEqual(parse_temperature("METAR KLAX 031053Z 10003KT 10SM FEW250 16/11 A3005 RMK AO2 SLP174 T01560106 $"), 156)
        self.assertEqual(parse_temperature("METAR LFPG 031100Z COR 06011KT 9000 BKN023 08/06 Q1026 NOSIG"), 80)
        self.assertEqual(parse_temperature("EGLL 031050Z AUTO 04011KT 350V060 9999 SCT045 M11/M17 Q1032"), -110)
        self.assertEqual(parse_temperature("KDEN 040153Z 00000KT 10SM CLR 02/M04 A3012 RMK AO2 T10171022"), -17)
        self.assertEqual(parse_temperature("ZGSZ 040000Z 02004MPS 9999 NSC 24/// Q1018"), 240)
        self.assertIsNone(parse_temperature("METAR LFPB 040100Z 25005KT CAVOK Q1020"))
        self.assertTrue(is_special("SPECI KLAX 031120Z 10003KT 16/11"))
        self.assertFalse(is_special("KLAX 031120Z 10003KT 16/11"))
        self.assertTrue(is_special("METAR KLAX 031120Z 16/11", "SPECI"))

    def test_rounding_as_the_rule_says(self):
        for tenths, want in ((139, 14), (174, 17), (165, 17), (-17, -2), (-85, -8), (80, 8)):
            self.assertEqual(whole_degrees(tenths, "C"), want, tenths)
        for tenths, want in ((183, 65), (178, 64), (190, 66), (-225, -8), (175, 64)):
            self.assertEqual(whole_degrees(tenths, "F"), want, tenths)


class CollectorTests(unittest.TestCase):
    def test_resolution_source(self):
        d = ("… hourly temperature remark … The resolution source for this market will be the routine observations for "
             "Paris-Le Bourget Airport (LFPB) published here:\nhttps://aviationweather.gov/api/data/metar?ids=LFPB&format=raw&date=2026-11-05T02:00:00Z&hours=30")
        icao, read_at, rule = resolution_source(d)
        self.assertEqual((icao, rule), ("LFPB", True))
        self.assertEqual(read_at, datetime(2026, 11, 5, 2, 0, tzinfo=timezone.utc))
        icao, read_at, rule = resolution_source("…recorded by the Hong Kong Observatory… https://www.weather.gov.hk/en/cis/climat.htm")
        self.assertEqual((icao, read_at, rule), (None, None, False))
        self.assertEqual(resolution_source("reported at Denver International Airport (KDEN) on 4 Nov")[0], "KDEN")

    def test_day_summary_uses_local_dates_and_skips_specials(self):
        day = date(2026, 11, 4)
        reports = [rep("22:30", 90, day="2026-11-03"), rep("00:00", 85), rep("01:00", 80), rep("11:00", 150), rep("11:20", 170, special=True),
                   rep("12:00", 160), rep("13:00", 162), rep("14:00", 150), rep("18:00", 120), rep("23:30", 100), rep("23:00", 101, day="2026-11-04")]
        s = day_summary(reports, PARIS, day, until=datetime(2026, 11, 5, 0, 0, tzinfo=PARIS))
        self.assertEqual(s.n, 7)             # 22:30 UTC on the 3rd is the 3rd in Paris, 23:00 UTC on the 4th is the 5th; the SPECI is skipped
        self.assertEqual(s.max_tenths, 162)
        self.assertEqual(s.max_at.strftime("%H:%M"), "14:00")
        self.assertEqual(s.min_tenths, 80)
        self.assertEqual(s.latest_tenths, 120)
        self.assertAlmostEqual(s.gap_min, 600)   # 01:00 → 11:00 UTC

    def test_lock_rules(self):
        day = date(2026, 11, 4)
        reports = [rep(f"{h:02d}:00", t) for h, t in zip(range(0, 22), [80, 78, 77, 76, 76, 75, 76, 80, 95, 110, 130, 150, 160, 165, 162, 158, 150, 140, 130, 120, 112, 105])]
        at = lambda h, m=0, d=4: datetime(2026, 11, d, h, m, tzinfo=PARIS)  # noqa: E731
        s = day_summary(reports, PARIS, day, until=at(19, 30))
        self.assertEqual(lock_state("highest", s, at(19, 30), 20, 10, 180), (False, "waiting for 20:00"))
        s = day_summary(reports, PARIS, day, until=at(20, 10))
        ok, why = lock_state("highest", s, at(20, 10), 20, 10, 180)
        self.assertTrue(ok, why)
        self.assertEqual(lock_state("lowest", s, at(20, 10), 20, 10, 180), (False, "lows wait for the day to end"))
        # the reading is still at the high: not yet
        hot = reports + [rep("20:00", 165)]
        s = day_summary(hot, PARIS, day, until=at(21, 30))
        self.assertEqual(lock_state("highest", s, at(21, 30), 20, 10, 180)[0], False)
        # a long silence from the station blocks it, the day being over unblocks everything but a gap
        s = day_summary(reports, PARIS, day, until=at(23, 59))
        self.assertIn("without a report", lock_state("highest", s, at(23, 59), 20, 10, 60)[1])
        s = day_summary(reports + [rep("22:30", 100)], PARIS, day, until=at(1, 0, d=5))
        self.assertEqual(lock_state("lowest", s, at(1, 0, d=5), 20, 10, 180), (True, "day over"))
        self.assertEqual(lock_state("highest", DaySummaryEmpty(), at(1, 0, d=5), 20, 10, 180), (False, "no reports yet"))

    def test_winner_and_payers(self):
        mk = lambda lo, hi: Bucket("highest", "Paris", "C", lo, hi, date(2026, 11, 4), f"{lo}-{hi}")  # noqa: E731
        ladder = [mk(None, 9), mk(10, 10), mk(11, 11), mk(12, 12), mk(13, None)]
        self.assertEqual(winning_bucket(ladder, 115, "C").label, "12°C")     # 11.5 rounds up
        self.assertEqual(winning_bucket(ladder, 114, "C").label, "11°C")
        self.assertEqual(winning_bucket(ladder, 170, "C").label, "13°C or higher")
        us = [Bucket("highest", "Denver", "F", None, 47, date(2026, 11, 4), ""), Bucket("highest", "Denver", "F", 48, 49, date(2026, 11, 4), ""),
              Bucket("highest", "Denver", "F", 50, None, date(2026, 11, 4), "")]
        self.assertEqual(winning_bucket(us, 90, "F").label, "48–49°F")        # 9.0°C = 48.2°F
        w = winning_bucket(ladder, 115, "C")
        p = payers(ladder, w)
        self.assertEqual([(b.label, o) for b, o in p], [("12°C", "Yes"), ("9°C or below", "No"), ("10°C", "No"), ("11°C", "No"), ("13°C or higher", "No")])


def DaySummaryEmpty():
    from polybot.collector import DaySummary
    return DaySummary(day=date(2026, 11, 4))


if __name__ == "__main__":
    unittest.main()


class AfternoonTests(unittest.TestCase):
    def test_window(self):
        from polybot.collector import in_window
        at = lambda h, m: datetime(2026, 11, 4, h, m, tzinfo=PARIS)  # noqa: E731
        self.assertEqual(in_window(at(14, 59), 15), -1)
        self.assertEqual(in_window(at(15, 0), 15), 0)
        self.assertEqual(in_window(at(15, 59), 15), 0)
        self.assertEqual(in_window(at(16, 0), 15), 1)
