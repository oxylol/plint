"""What a call costs, in dollars per million tokens. Read off platform.claude.com/docs/en/about-claude/pricing
on 2026-11-14. The spend meter is only as right as this table: when Anthropic changes a price or the agent
moves to another model, change it here."""
from __future__ import annotations

# model id: (input, output, cache read, cache write 5 min)
PER_MTOK: dict[str, tuple[float, float, float, float]] = {
    "claude-fable-5-2": (8.00, 40.00, 0.20, 10.00),
    "claude-fable-5-1": (10.00, 50.00, 0.25, 12.50),
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-opus-5": (5.00, 25.00, 0.50, 6.25),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10, 2.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),      # prompts up to 100,000 tokens
}
HAIKU_55_LONG = (0.50, 2.50, 0.05, 0.625)                # the same model above 100,000 prompt tokens
WEB_SEARCH_USD = 10.0 / 1000                              # per search; web fetch costs tokens only
BATCH_FACTOR = 0.5                                        # the Batches API bills half of everything


def cost_usd(model: str, in_tok: int = 0, out_tok: int = 0, cache_read: int = 0, cache_write: int = 0,
             searches: int = 0, batch: bool = False) -> float:
    """Dollars for one response's usage. An unknown model is billed as the dearest in the table, so a
    typo can only make the meter read high."""
    rates = PER_MTOK.get(model) or max(PER_MTOK.values())
    if model == "claude-haiku-5-5" and in_tok + cache_read + cache_write > 100_000:
        rates = HAIKU_55_LONG
    i, o, cr, cw = rates
    tokens = (in_tok * i + out_tok * o + cache_read * cr + cache_write * cw) / 1e6
    return tokens * (BATCH_FACTOR if batch else 1.0) + searches * WEB_SEARCH_USD
