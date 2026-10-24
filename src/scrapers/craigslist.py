"""Scrape used-car listings from Craigslist "cars & trucks" across several metro areas.

Step 1: keyword-search each area for each target model and keep only result titles that
        name a target make/model and an in-range year (dealer spam matches keywords loosely).
Step 2: open each kept posting and parse its attribute table, location, and description.
"""
import json
import re
from itertools import zip_longest
from urllib.parse import quote_plus

from bs4 import BeautifulSoup

from src.config import CRAIGSLIST_AREAS, MAKE_ALIASES, MAX_YEAR, MIN_YEAR, OTHER_MAKES_RE, TARGETS
from src.scrapers.utils import BlockedError, PoliteFetcher, save_csv

SEARCH_URL = "https://www.craigslist.org/search/area/{area}?cat=cta&query={query}"
PER_MODEL = 75  # postings to fetch per model

YEAR_RE = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")
SKIP_RE = re.compile(r"\b(wanted|parting out|part out|parts only|for parts|lease|takeover|rent(al)?|wtb)\b", re.I)


def in_range(year):
    return MIN_YEAR <= int(year) <= MAX_YEAR


def match_title(title, make, pattern):
    """True if a search-result title is plausibly this make/model in the target year range.
    Titles with no year pass; the posting's year attribute is checked after fetching."""
    text = title.lower()
    if SKIP_RE.search(text) or not re.search(pattern, text):
        return False
    if not re.search(rf"\b({MAKE_ALIASES[make]})\b", text) and OTHER_MAKES_RE.search(text):
        return False
    year = YEAR_RE.search(text)
    return not year or in_range(year.group(1))


def parse_search_page(html):
    soup = BeautifulSoup(html or "", "lxml")
    return [{"title": li.get("title", ""), "url": li.a["href"]}
            for li in soup.select("li.cl-static-search-result") if li.a and li.a.get("href")]


def text_of(soup, selector):
    el = soup.select_one(selector)
    return el.get_text(" ", strip=True) if el else None


def parse_posting(html, url):
    soup = BeautifulSoup(html, "lxml")
    row = {
        "url": url,
        "title": text_of(soup, "#titletextonly"),
        "price": text_of(soup, ".postingtitletext .price"),
    }
    for info in soup.select("p.postinginfo"):
        label = info.get_text(strip=True).split(":", 1)[0]
        if label == "post id":
            row["post_id"] = info.get_text(strip=True).split(":", 1)[1].strip()
        elif label in ("posted", "updated") and info.time:
            row[f"{label}_at"] = info.time.get("datetime")

    # Attribute table: the first group holds year + make/model, the rest are "label: value" pairs.
    for attr in soup.select(".attrgroup .attr"):
        label, value = attr.select_one(".labl"), attr.select_one(".valu")
        if label and value:
            row[label.get_text(strip=True).rstrip(":").strip().lower().replace(" ", "_")] = value.get_text(" ", strip=True)
    row["year_attr"] = text_of(soup, ".attr.important .valu.year")
    row["makemodel_attr"] = text_of(soup, ".attr.important .valu.makemodel")

    body = soup.select_one("#postingbody")
    if body:
        for junk in body.select(".print-information"):
            junk.decompose()
        row["description"] = body.get_text(" ", strip=True)

    category = soup.select_one(".crumb.category a")
    if category and "cat=" in category.get("href", ""):
        cat = category["href"].split("cat=")[1][:3]
        row["seller_type"] = {"cto": "owner", "ctd": "dealer"}.get(cat, cat)
    row["area"] = text_of(soup, ".crumb.area a")

    ld = soup.select_one("#ld_posting_data")
    if ld:
        place = json.loads(ld.string).get("offers", {}).get("availableAtOrFrom", {})
        address, geo = place.get("address", {}), place.get("geo", {})
        row.update(city=address.get("addressLocality"), state=address.get("addressRegion"),
                   zip=address.get("postalCode"), latitude=geo.get("latitude"), longitude=geo.get("longitude"))
    return row


def candidates_for(fetcher, make, cfg):
    """Search every area for one model; return matching results interleaved across areas
    so the sample is spread across regions instead of filled by the first city."""
    per_area = [[r for r in parse_search_page(fetcher.get(SEARCH_URL.format(area=area, query=quote_plus(cfg["cl_query"]))))
                 if match_title(r["title"], make, cfg["pattern"])]
                for area in CRAIGSLIST_AREAS]
    return [r for group in zip_longest(*per_area) for r in group if r]


def main():
    fetcher = PoliteFetcher("craigslist")
    rows, seen = [], set()
    try:
        for make, models in TARGETS.items():
            for model, cfg in models.items():
                candidates = candidates_for(fetcher, make, cfg)
                kept = 0
                for cand in candidates:
                    post_key = cand["url"].rstrip("/").rsplit("/", 1)[-1]
                    if kept >= PER_MODEL or post_key in seen:
                        continue
                    seen.add(post_key)
                    html = fetcher.get(cand["url"])
                    row = parse_posting(html, cand["url"]) if html else {}
                    if (row.get("year_attr") or "").isdigit() and in_range(row["year_attr"]):
                        rows.append({**row, "make": make, "search_model": model, "search_title": cand["title"]})
                        kept += 1
                print(f"{make} {model}: {len(candidates)} matching titles, kept {kept}")
                save_csv(rows, "craigslist", source="craigslist")  # checkpoint: partial results visible
    except BlockedError as exc:
        print(f"BLOCKED: {exc}. Saving partial results.")
    save_csv(rows, "craigslist", source="craigslist")
    print(f"Saved {len(rows)} listings")


if __name__ == "__main__":
    main()
