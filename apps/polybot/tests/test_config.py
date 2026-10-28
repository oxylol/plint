import unittest

from polybot.config import parse_variants


class VariantTests(unittest.TestCase):
    def test_parse(self):
        v = parse_variants("month:window=month; day : window=day, min_agree=4 ;all:window=all;week4:POLY_MIN_AGREE=4,label=Week 4")
        self.assertEqual([n for n, _ in v], ["month", "day", "all", "week4"])
        self.assertEqual(v[1][1], {"window": "day", "min_agree": "4"})
        self.assertEqual(v[3][1], {"min_agree": "4", "label": "Week 4"})
        self.assertEqual(parse_variants(""), [])
        self.assertEqual(parse_variants("plain"), [("plain", {})])

    def test_rejects_bad_input(self):
        for raw in ("consensus:window=day", "a:window=day;a:window=month", "x:nope=1", "y:window", "bad name:window=day"):
            with self.assertRaises(ValueError, msg=raw):
                parse_variants(raw)


if __name__ == "__main__":
    unittest.main()
