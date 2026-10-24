"""Shared project settings: target vehicles, scrape regions, and file paths."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
HTML_CACHE_DIR = RAW_DIR / "html"
OUT_DIR = ROOT / "out"

MIN_YEAR = 2012
MAX_YEAR = 2024

# make -> model -> settings
#   truecar:  TrueCar URL slug (/used-cars-for-sale/listings/<make>/<slug>/)
#   cl_query: Craigslist keyword search (the auto_make_model filter misses most posts)
#   pattern:  regex a listing title must match (after the make) to count as this model
TARGETS = {
    "Toyota": {
        "Camry": {"truecar": "camry", "cl_query": "camry", "pattern": r"camry"},
        "Corolla": {"truecar": "corolla", "cl_query": "corolla", "pattern": r"corolla(?!\s*cross)"},
        "RAV4": {"truecar": "rav4", "cl_query": "rav4", "pattern": r"rav\s*-?\s*4"},
        "Tacoma": {"truecar": "tacoma", "cl_query": "tacoma", "pattern": r"tacoma"},
    },
    "Honda": {
        "Civic": {"truecar": "civic", "cl_query": "honda civic", "pattern": r"civic"},
        "Accord": {"truecar": "accord", "cl_query": "honda accord", "pattern": r"accord(?!\s*crosstour)"},
        "CR-V": {"truecar": "cr-v", "cl_query": "crv", "pattern": r"cr\s*-?\s*v"},
    },
    "Ford": {
        "F-150": {"truecar": "f-150", "cl_query": "f150", "pattern": r"f\s*-?\s*150"},
        "Escape": {"truecar": "escape", "cl_query": "ford escape", "pattern": r"escape"},
        "Explorer": {"truecar": "explorer", "cl_query": "ford explorer", "pattern": r"explorer(?!\s*sport\s*trac)"},
    },
    "Chevrolet": {
        "Silverado 1500": {"truecar": "silverado-1500", "cl_query": "silverado 1500", "pattern": r"silverado\s*-?\s*1500"},
        "Equinox": {"truecar": "equinox", "cl_query": "equinox", "pattern": r"equinox"},
        "Malibu": {"truecar": "malibu", "cl_query": "malibu", "pattern": r"malibu"},
    },
}

# Alternate spellings sellers use for each make in free-text titles
MAKE_ALIASES = {
    "Toyota": r"toyota",
    "Honda": r"honda",
    "Ford": r"ford",
    "Chevrolet": r"chevrolet|chevy|chev",
}

# Non-target makes that keyword searches drag in ("Ford Escape vs Mercury Mariner", dealer lots, etc.)
OTHER_MAKES_RE = re.compile(
    r"\b(nissan|hyundai|kia|mazda|subaru|jeep|dodge|ram|gmc|buick|cadillac|lexus|acura|infiniti|bmw|audi|"
    r"mercedes|volkswagen|vw|volvo|land rover|mini|mitsubishi|lincoln|mercury|chrysler|tesla|scion|pontiac|saturn)\b",
    re.I,
)

# Craigslist area slugs (www.craigslist.org/search/area/<slug>), spread across regions
CRAIGSLIST_AREAS = [
    "philadelphia", "pittsburgh", "newyork", "boston", "washingtondc", "atlanta",
    "miami", "chicago", "detroit", "dallas", "houston", "denver",
    "phoenix", "losangeles", "sfbay", "seattle",
]
