"""A rough market type from a title and slug, shared by the research scripts and the bot."""
from __future__ import annotations

SPORTS_HINTS = (" vs. ", " vs ", "Spread:", "O/U", "Moneyline", "(BO3)", "(BO5)", "(BO7)")
SPORTS_PREFIX = {"nba", "nfl", "nhl", "mlb", "epl", "ucl", "uel", "lal", "sea", "bun", "fl1", "mls", "atp", "wta", "cfb", "cbb",
                 "cs2", "val", "dota2", "lol", "hok", "ufc", "f1", "nrl", "afl", "cri", "ipl", "kbo", "npb", "mex", "nor", "jap",
                 "ja2", "ere", "el1", "elc", "chi", "itsb", "kor", "ahl", "col1", "azec", "acle", "gtm", "spl", "bel", "tur",
                 "por", "ned", "sco", "arg", "bra", "usl", "wnba", "ncaab", "ncaaf", "pga", "golf", "tennis", "box", "mma"}
KEYWORDS = [
    ("crypto", ("bitcoin", "btc", "ethereum", "eth ", "solana", "crypto", "fdv", "token", "coin", "xrp", "doge")),
    ("politics", ("election", "senate", "governor", "president", "mayor", "prime minister", "seats", "party", "house seat",
                  "congress", "parliament", "mandate", "nominee", "primary", "vote", "minister", "coalition", "likud", "chancellor")),
    ("economics", ("fed ", "interest rate", "rate cut", "gdp", "inflation", "cpi", "unemployment", "tariff", "oil", "wti", "gold", "recession")),
    ("tech", ("openai", "gpt", "ai model", "grok", "gemini", "claude", "apple", "tesla", "spacex", "launch", "iphone")),
    ("geopolitics", ("ceasefire", "war", "invade", "strike", "missile", "troops", "nato", "hostage", "ukraine", "russia", "iran",
                     "israel", "gaza", "china", "taiwan", "sanction")),
    ("weather", ("temperature", "weather", "hurricane", "rain", "snow", "°")),
    ("culture", ("tweet", "post", "mention", "album", "movie", "box office", "grammy", "oscar", "rotten", "song", "netflix")),
]


def category(title: str | None, slug: str | None) -> str:
    slug = slug or ""
    title = title or ""
    if slug.split("-")[0].lower() in SPORTS_PREFIX or any(h in title for h in SPORTS_HINTS):
        return "sports"
    low = (title + " " + slug).lower()
    for name, words in KEYWORDS:
        if any(k in low for k in words):
            return name
    return "other"
