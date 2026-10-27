"""Push notifications through Plint's ntfy (same topic as `plint notify`)."""
from __future__ import annotations

import logging
import urllib.request

log = logging.getLogger("polybot.notify")


class Notifier:
    def __init__(self, url: str, token: str = ""):
        self.url = url
        self.token = token

    def send(self, title: str, body: str, priority: str = "default", tags: str = "chart_with_upwards_trend") -> None:
        if not self.url:
            return
        headers = {"Title": title.encode("ascii", "replace").decode(), "Priority": priority, "Tags": tags}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.url, data=body.encode("utf-8"), headers=headers, method="POST")
        try:
            urllib.request.urlopen(req, timeout=10).close()
        except Exception as e:
            log.warning("ntfy failed: %s", e)
