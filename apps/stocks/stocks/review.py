"""The agent's weekly look at its own record: closed trades and settled forecasts in, a rewritten playbook out."""
from __future__ import annotations

from .bars import d2s
from .pipeline import Lab, books


def weekly(lab: Lab) -> str:
    cfg = lab.cfg
    if not cfg.has_agent:
        return "no key: nothing to review"
    last = lab.bars.last_day(cfg.index_symbol)
    if not last or not lab.store.since("agent"):
        return "the agent has no record yet"
    from .agent.data import LiveDeskData
    from .agent.desk import Desk
    panel = lab.live_panel(last)
    rec = books(lab, panel).get("agent")
    closed = []
    if rec is not None and rec.book:
        bought: dict[str, tuple[str, str]] = {}
        for f in rec.book.fills:
            if f.side == "buy":
                bought.setdefault(f.sym, (d2s(f.d), f.why))
            elif f.ret is not None:
                since, why = bought.pop(f.sym, ("", ""))
                closed.append({"symbol": f.sym, "bought": since, "sold": d2s(f.d), "ret": round(f.ret, 4),
                               "why_bought": why, "why_sold": f.why})
    scored = lab.store.signals("agent", settled=True, limit=400)
    frame = lab.assets.frame()
    data = LiveDeskData(panel, cfg.index_symbol, lab.assets.names(), lab.universe.ranks(last),
                        exchanges=dict(zip(frame["sym"], frame["exchange"])) if len(frame) else {})
    text = Desk(cfg, lab.store).weekly_review(data, closed[-60:], scored)
    lab.say(f"the agent reviewed {len(closed)} closed trades and {len(scored)} settled forecasts and rewrote its playbook")
    return text
