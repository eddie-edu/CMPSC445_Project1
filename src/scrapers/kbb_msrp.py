"""Collect original MSRP for every target make/model/year from Kelley Blue Book.

Each KBB model-year page (kbb.com/<make>/<model>/<year>/) embeds its trim list as JSON with each
trim's original MSRP. Saves msrp_trims.csv (one row per trim) and msrp.csv (one row per
make/model/year; base = cheapest trim, the usual meaning of "MSRP from $X").
"""
from src.config import MAX_YEAR, MIN_YEAR, TARGETS
from src.scrapers.utils import BlockedError, PoliteFetcher, next_data, save_csv

PAGE_URL = "https://www.kbb.com/{make}/{model}/{year}/"


def parse_trims(html):
    """[(trim name, original MSRP)] from the page's trimsData JSON (found by depth-first search)."""
    stack = [next_data(html)]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if "trimsData" in node:
                return [(t["displayName"], t["originalMSRP"]) for t in node["trimsData"] or [] if t.get("originalMSRP")]
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return []


def main():
    fetcher = PoliteFetcher("kbb")
    rows = []
    try:
        for make, models in TARGETS.items():
            for model, cfg in models.items():
                years = 0
                for year in range(MIN_YEAR, MAX_YEAR + 1):
                    url = PAGE_URL.format(make=make.lower(), model=cfg["truecar"], year=year)
                    trims = parse_trims(fetcher.get(url))
                    rows += [{"make": make, "model": model, "year": year, "trim": trim, "msrp": msrp,
                              "source_url": url} for trim, msrp in trims]
                    years += bool(trims)
                print(f"{make} {model}: MSRP for {years} of {MAX_YEAR - MIN_YEAR + 1} years")
    except BlockedError as exc:
        print(f"BLOCKED: {exc}. Saving partial results.")

    trims = save_csv(rows, "msrp_trims")
    per_year = (trims.groupby(["make", "model", "year", "source_url"], as_index=False)["msrp"]
                .agg(base_msrp="min", max_msrp="max", n_trims="size"))
    save_csv(per_year, "msrp")
    print(f"Saved {len(per_year)} make/model/year rows and {len(trims)} trim rows")


if __name__ == "__main__":
    main()
