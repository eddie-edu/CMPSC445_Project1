"""Scrape used-car listings from TrueCar search pages.

Each search page embeds its 30 results as JSON (Next.js Apollo cache), so no detail pages are needed.
"""
from src.config import MAX_YEAR, MIN_YEAR, TARGETS
from src.scrapers.utils import BlockedError, PoliteFetcher, next_data, save_csv

SEARCH_URL = "https://www.truecar.com/used-cars-for-sale/listings/{make}/{model}/"
PAGES_PER_MODEL = 4  # 30 listings per page


def parse_search_page(html):
    """Return (listing dicts, total result count) from one search page."""
    data = next_data(html)
    if not data:
        return [], 0
    state = data["props"]["pageProps"]["__APOLLO_STATE__"]
    # The organic results query; the other marketplaceListingSearch entry holds sponsored ads.
    key = next((k for k in state["ROOT_QUERY"]
                if k.startswith("marketplaceListingSearch") and '"sponsored":true' not in k), None)
    if not key:
        return [], 0
    search = state["ROOT_QUERY"][key]
    return [state[edge["node"]["__ref"]] for edge in search["edges"]], search.get("totalCount", 0)


def flatten(listing):
    vehicle = listing["vehicle"]
    details = vehicle.get("details") or {}
    style = vehicle.get("style") or {}
    mpg = vehicle.get("mpg") or {}
    history = listing.get("conditionHistory") or {}
    return {
        "vin": vehicle.get("vin"),
        "price": (listing.get("pricing") or {}).get("listPrice"),
        "year": vehicle.get("year"),
        "make": vehicle["make"]["name"],
        "model": vehicle["model"]["name"],
        "trim": style.get("trimName"),
        "style": style.get("name"),
        "mileage": vehicle.get("mileage"),
        "body_style": vehicle.get("bodyStyle"),
        "drivetrain": details.get("driveTypeAcronym") or style.get("driveType"),
        "engine": vehicle.get("engine"),
        "fuel_type": vehicle.get("fuelType"),
        "transmission": vehicle.get("transmission"),
        "exterior_color": vehicle.get("exteriorColor"),
        "exterior_color_name": details.get("exteriorColorName"),
        "interior_color": vehicle.get("interiorColor"),
        "mpg_city": mpg.get("city"),
        "mpg_highway": mpg.get("highway"),
        "mpg_combined": mpg.get("combined"),
        "certified_pre_owned": vehicle.get("certifiedPreOwned"),
        "accident_count": history.get("accidentCount"),
        "owner_count": history.get("ownerCount"),
        "clean_title": history.get("isCleanTitle"),
        "price_rating": (listing.get("marketAnalysis") or {}).get("priceQuality"),
        "key_features": "; ".join(vehicle.get("keyFeatures") or []),
        "city": details.get("dealerCity"),
        "state": details.get("dealerState"),
        "listed_at": details.get("listedAt"),
        "seller_type": "dealer",
        "url": f"https://www.truecar.com/used-cars-for-sale/listing/{vehicle.get('vin')}/",
    }


def main():
    fetcher = PoliteFetcher("truecar")
    rows, seen = [], set()
    try:
        for make, models in TARGETS.items():
            for model, cfg in models.items():
                kept = 0
                for page in range(1, PAGES_PER_MODEL + 1):
                    url = SEARCH_URL.format(make=make.lower(), model=cfg["truecar"])
                    listings, total = parse_search_page(fetcher.get(url + (f"?page={page}" if page > 1 else "")))
                    for row in map(flatten, listings):
                        if MIN_YEAR <= (row["year"] or 0) <= MAX_YEAR and row["vin"] not in seen:
                            seen.add(row["vin"])
                            rows.append({**row, "search_model": model})
                            kept += 1
                    if page * 30 >= total:
                        break
                print(f"{make} {model}: {kept} listings in {MIN_YEAR}-{MAX_YEAR}")
    except BlockedError as exc:
        print(f"BLOCKED: {exc}. Saving partial results.")
    save_csv(rows, "truecar", source="truecar")
    print(f"Saved {len(rows)} unique listings")


if __name__ == "__main__":
    main()
