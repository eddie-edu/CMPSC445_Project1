"""Shared scraping helpers: polite cached fetching, Next.js page data, and CSV output."""
import hashlib
import json
import random
import re
import time
import urllib.robotparser
from datetime import datetime, timezone
from urllib.parse import urlsplit

import pandas as pd
from curl_cffi import requests

from src.config import HTML_CACHE_DIR, RAW_DIR


class BlockedError(RuntimeError):
    """The site refused the request (403/429)."""


class PoliteFetcher:
    """Fetches pages with a Chrome TLS fingerprint, 1.5-3.5 s apart, honoring robots.txt.
    Every page is cached under data/raw/html/<source>/, so re-runs don't hit the site again
    (delete that folder to re-download)."""

    def __init__(self, source):
        self.cache_dir = HTML_CACHE_DIR / source
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session(impersonate="chrome")
        self.robots = {}
        self.last_request = 0.0

    def allowed(self, url):
        base = "{0.scheme}://{0.netloc}".format(urlsplit(url))
        if base not in self.robots:
            self.robots[base] = urllib.robotparser.RobotFileParser(base + "/robots.txt")
            self.robots[base].parse(self.session.get(base + "/robots.txt").text.splitlines())
        return self.robots[base].can_fetch("*", url)

    def get(self, url):
        """Page HTML (cached), or None if missing. Raises BlockedError if the site refuses."""
        path = self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".html")
        if path.exists():
            return path.read_text(encoding="utf-8")
        if not self.allowed(url):
            raise PermissionError(f"robots.txt disallows {url}")
        for attempt in range(3):
            time.sleep(max(0, self.last_request + random.uniform(1.5, 3.5) - time.time()))
            self.last_request = time.time()
            try:
                resp = self.session.get(url, timeout=30)
            except requests.RequestsError:
                continue
            if resp.status_code in (403, 429):
                raise BlockedError(f"HTTP {resp.status_code} for {url}")
            if resp.ok:
                path.write_text(resp.text, encoding="utf-8")
                return resp.text
            if resp.status_code == 404:
                return None
        return None


def next_data(html):
    """The JSON a Next.js page embeds in <script id="__NEXT_DATA__">, or None."""
    match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html or "", re.S)
    return json.loads(match.group(1)) if match else None


def save_csv(rows, name, **constant_columns):
    """Write rows to data/raw/<name>.csv with any constant columns plus a scraped_at timestamp."""
    df = pd.DataFrame(rows).assign(**constant_columns,
                                   scraped_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / f"{name}.csv", index=False)
    return df
