"""Build a fresh AG News-style test set from news RSS feeds, for a contamination probe.

Only articles published after CUTOFF are kept, so no model released before then can have trained
on them. Labels start from the feed's section (World / Business / Sci/Tech / Sports) and are then
checked by hand (`verified` column). Article text is kept locally only (data/fresh/*.jsonl is
gitignored); the committed file lists URL, date, source and label, and this script can rebuild
the text.
"""
from __future__ import annotations

import html
import json
import random
import re
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path
from datetime import datetime, timezone

CUTOFF = datetime(2026, 9, 16, tzinfo=timezone.utc)  # day after Jev's public launch
PER_CLASS = 50
FEEDS = {
    "World": ["https://feeds.bbci.co.uk/news/world/rss.xml", "https://www.theguardian.com/world/rss",
              "https://feeds.npr.org/1004/rss.xml"],
    "Business": ["https://feeds.bbci.co.uk/news/business/rss.xml", "https://feeds.npr.org/1006/rss.xml"],
    "Sci/Tech": ["https://feeds.bbci.co.uk/news/technology/rss.xml",
                 "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
                 "https://www.theguardian.com/science/rss", "https://feeds.npr.org/1019/rss.xml",
                 "https://feeds.npr.org/1007/rss.xml"],
    "Sports": ["https://feeds.bbci.co.uk/sport/rss.xml", "https://feeds.npr.org/1055/rss.xml"],
}
OUT = Path(__file__).resolve().parents[1] / "data" / "fresh"


def clean(text: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    return re.sub(r"\s+", " ", text).strip()


def items(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    root = ET.fromstring(urllib.request.urlopen(req, timeout=30).read())
    source = re.sub(r"^https?://(www\.|feeds\.)?", "", url).split("/")[0]
    for it in root.iter("item"):
        title, desc = clean(it.findtext("title")), clean(it.findtext("description"))
        link, date = (it.findtext("link") or "").strip(), it.findtext("pubDate")
        if not (title and link and date):
            continue
        try:
            when = parsedate_to_datetime(date)
        except (TypeError, ValueError):
            continue
        yield {"title": title, "description": desc[:400], "url": link, "date": when.isoformat(), "source": source,
               "_when": when}


def main() -> None:
    rng = random.Random(20260925)
    rows, seen = [], set()
    for label, urls in FEEDS.items():
        pool = []
        for url in urls:
            try:
                for r in items(url):
                    if r["_when"] >= CUTOFF and r["url"] not in seen and r["title"] not in seen:
                        seen.update([r["url"], r["title"]])
                        pool.append(r)
            except Exception as e:  # a dead feed shouldn't stop the others
                print(f"  skip {url}: {e}")
        rng.shuffle(pool)
        picked = pool[:PER_CLASS]
        print(f"{label}: {len(pool)} after cutoff, kept {len(picked)}")
        for r in picked:
            r.pop("_when")
            rows.append({**r, "label": label, "label_from": "feed section", "verified": None})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "ag_news_fresh.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with (OUT / "ag_news_fresh_index.jsonl").open("w") as f:  # committed: no article text
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("url", "date", "source", "label")}) + "\n")
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
