"""Strict checks of generated HTML: well-formed (every non-void element closed in order), self-contained
(no scripts, no external resources), and every in-page link pointing at an element that exists."""

from __future__ import annotations

import re
from html.parser import HTMLParser

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


class Checker(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.ids: list[str] = []
        self.links: list[str] = []
        self.sources: list[str] = []
        self.tags: dict[str, int] = {}
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags[tag] = self.tags.get(tag, 0) + 1
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.append(attributes["id"])
        if tag == "a" and "href" in attributes:
            self.links.append(attributes["href"])
        for key in ("src", "href"):
            if key in attributes and tag != "a":
                self.sources.append(attributes[key])
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.stack.pop()

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closes <{self.stack[-1] if self.stack else 'nothing'}>")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
            return
        self.stack.pop()

    def handle_data(self, data):
        self.text.append(data)


def check(text: str) -> Checker:
    checker = Checker()
    checker.feed(text)
    checker.close()
    assert not checker.errors, checker.errors[:5]
    assert not checker.stack, f"unclosed elements: {checker.stack[-5:]}"
    assert len(checker.ids) == len(set(checker.ids)), "duplicate element ids"
    missing = [href for href in checker.links if href.startswith("#") and href[1:] not in set(checker.ids)]
    assert not missing, f"links to missing anchors: {missing[:5]}"
    assert "script" not in checker.tags, "the page must not run scripts"
    external = [s for s in checker.sources if re.match(r"^(https?:)?//", s) or not s.startswith("data:")]
    assert not external, f"external resources: {external[:3]}"
    return checker
