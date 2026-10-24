"""Part 2: integrate the TrueCar and Craigslist scrapes into one clean dataset.

Steps
  1. Map each source onto one schema (names, formats, units, category labels).
  2. Drop rows missing an important feature: price, year, make, model, mileage.
  3. Drop rows with bad values (placeholder prices, impossible mileage, wrong vehicle).
  4. Drop duplicates (VIN, then make/model/year/mileage, then Craigslist reposts).
  5. Merge KBB MSRP; write the dataset, a cleaning log, and attribute statistics to out/.
"""
import re

import numpy as np
import pandas as pd

from src.config import MAKE_ALIASES, OTHER_MAKES_RE, OUT_DIR, RAW_DIR, TARGETS

IMPORTANT = ["price", "year", "make", "model", "mileage"]
REFERENCE_YEAR = 2026  # year the listings were scraped

# Unified schema, in output order. A column one source lacks stays NaN for its rows.
COLUMNS = [
    "listing_id", "source", "vin", "price", "year", "make", "model", "trim", "mileage",
    "body_type", "drivetrain", "transmission", "fuel_type", "cylinders", "exterior_color",
    "title_status", "condition", "seller_type", "city", "state", "listing_date",
    "owner_count", "accident_count", "certified_pre_owned", "mpg_combined", "interior_color",
    "description", "key_features", "url",
]
CATEGORICAL = ["trim", "body_type", "drivetrain", "transmission", "fuel_type", "exterior_color",
               "title_status", "condition", "seller_type", "state"]

# Models of the target makes (and other makes) that keyword stuffing drags into searches
OTHER_MODELS_RE = re.compile(
    r"\b(tundra|sienna|highlander|4runner|prius|sequoia|venza|colorado|cruze|impala|traverse|tahoe|"
    r"suburban|trax|blazer|transit|edge|focus|fusion|ranger|expedition|bronco|mustang|pilot|odyssey|"
    r"fit|hr-?v|ridgeline|passport|rogue|murano|pathfinder|altima)\b",
    re.I,
)
COLOR_MAP = {"grey": "gray", "custom": "other", "unknown": np.nan, "tan": "brown", "beige": "brown", "gold": "yellow"}
# Craigslist has no AWD option (sellers pick 4wd), so AWD and 4WD can't be told apart across sources.
DRIVE_MAP = {"awd": "awd/4wd", "4wd": "awd/4wd"}
BODY_MAP = {"pickup": "truck", "offroad": "suv", "wagon": "other", "van": "other"}


def to_number(series):
    return pd.to_numeric(series.astype(str).str.replace(r"[$,\s]", "", regex=True), errors="coerce")


def lower(series):
    return series.astype("string").str.strip().str.lower()


# ---------------------------------------------------------------- step 1: unify schema

def unify_truecar(df):
    return pd.DataFrame({
        "listing_id": "tc_" + df["vin"],
        "source": "truecar",
        "vin": df["vin"],
        "price": df["price"],
        "year": df["year"],
        "make": df["make"],
        "model": df["search_model"],
        "trim": df["trim"],
        "mileage": df["mileage"],
        "body_type": lower(df["body_style"]),
        "drivetrain": lower(df["drivetrain"]),
        "transmission": lower(df["transmission"]),
        "fuel_type": lower(df["fuel_type"]).replace({"plug-in hybrid": "hybrid"}),
        "cylinders": to_number(df["engine"].str.extract(r"(?:Inline|V)-(\d+)")[0]),
        "exterior_color": lower(df["exterior_color"]),
        "title_status": df["clean_title"].map({True: "clean", False: "branded"}),
        "seller_type": "dealer",
        "city": df["city"],
        "state": df["state"],
        "listing_date": pd.to_datetime(df["listed_at"], utc=True).dt.date,
        "owner_count": df["owner_count"],
        "accident_count": df["accident_count"],
        "certified_pre_owned": df["certified_pre_owned"],
        "mpg_combined": df["mpg_combined"],
        "interior_color": lower(df["interior_color"]),
        "key_features": df["key_features"],  # TrueCar has no free-text description
        "url": df["url"],
    })


def parse_trim(makemodel, make, model, known_trims):
    """Pull a trim (e.g. 'XLE', 'EX-L', 'TRD Sport') out of Craigslist's free-text make/model field."""
    if not isinstance(makemodel, str):
        return np.nan
    text = re.sub(rf"\b({MAKE_ALIASES[make]})\b", " ", makemodel.lower())
    text = " " + " ".join(re.sub(TARGETS[make][model]["pattern"], " ", text).split()) + " "
    for trim in sorted(known_trims, key=len, reverse=True):  # longest first: 'EX-L' before 'EX'
        if f" {trim.lower()} " in text:
            return trim
    return np.nan


def unify_craigslist(df, trims_by_model):
    return pd.DataFrame({
        "listing_id": "cl_" + df["post_id"],
        "source": "craigslist",
        "vin": df["vin"].str.upper().str.strip(),
        "price": to_number(df["price"]),
        "year": to_number(df["year_attr"]),
        "make": df["make"],
        "model": df["search_model"],
        "trim": [parse_trim(mm, mk, md, trims_by_model.get((mk, md), []))
                 for mm, mk, md in zip(df["makemodel_attr"], df["make"], df["search_model"])],
        "mileage": to_number(df["odometer"]),
        "body_type": lower(df["type"]),
        "drivetrain": lower(df["drive"]),
        "transmission": lower(df["transmission"]),
        "fuel_type": lower(df["fuel"]),
        "cylinders": to_number(df["cylinders"].str.extract(r"(\d+)")[0]),
        "exterior_color": lower(df["paint_color"]),
        "title_status": lower(df["title_status"]),
        "condition": lower(df["condition"]),
        "seller_type": df["seller_type"].replace({"owner": "private"}),
        "city": df["city"],
        "state": df["state"],
        "listing_date": pd.to_datetime(df["posted_at"], utc=True).dt.date,
        "description": df["description"],
        "url": df["url"],
        "_makemodel_attr": df["makemodel_attr"],  # used by the wrong-vehicle check, dropped after
    })


def harmonize(df):
    """Resolve label differences that remain after mapping."""
    df = df.replace({"body_type": BODY_MAP, "drivetrain": DRIVE_MAP, "exterior_color": COLOR_MAP})
    df.loc[~df["vin"].str.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", na=False), "vin"] = np.nan
    df[CATEGORICAL] = df[CATEGORICAL].fillna("unknown")
    return df


# ---------------------------------------------------------------- steps 2-4: drop rows

def age_of(df):
    return (REFERENCE_YEAR - df["year"] + 0.5).clip(lower=1)


def wrong_vehicle(row):
    """Craigslist post whose own make/model field names some other vehicle."""
    if row["source"] != "craigslist" or not isinstance(row["_makemodel_attr"], str):
        return False
    text = row["_makemodel_attr"].lower()
    if re.search(TARGETS[row["make"]][row["model"]]["pattern"], text):
        return False
    other_target_make = any(re.search(rf"\b({alias})\b", text)
                            for make, alias in MAKE_ALIASES.items() if make != row["make"])
    return bool(other_target_make or OTHER_MAKES_RE.search(text) or OTHER_MODELS_RE.search(text))


def price_outliers(df):
    """Log-price residual after a per-model fit on age and mileage, fenced at 3 IQRs. A plain per-model
    IQR fence would cut legitimately old or new cars. Salvage/rebuilt/branded titles are exempt:
    their price gap is real, not a data error."""
    resid = pd.Series(index=df.index, dtype=float)
    for _, g in df.groupby("model"):
        X = np.column_stack([np.ones(len(g)), REFERENCE_YEAR - g["year"], g["mileage"] / 10_000])
        y = np.log(g["price"])
        resid[g.index] = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    q1, q3 = resid.quantile([0.25, 0.75])
    fence = 3 * (q3 - q1)
    branded = df["title_status"].isin(["salvage", "rebuilt", "branded"])
    return ~branded & ((resid < q1 - fence) | (resid > q3 + fence))


def duplicates(df, keys, among=None):
    """Rows sharing `keys` with a more complete record (among rows where `among` is True)."""
    pool = (df if among is None else df[among]).dropna(subset=keys)
    completeness = pool.notna().sum(axis=1) + (pool != "unknown").sum(axis=1)
    pool = pool.loc[completeness.sort_values(ascending=False).index]
    return df.index.isin(pool[pool.duplicated(subset=keys)].index)


def clean(df):
    """Apply drop rules in order, logging how many rows each removed per source."""
    log = []

    def drop(step, mask):
        nonlocal df
        removed, df = df[mask], df[~mask]
        entry = {"step": step, "rows_left": len(df)}
        for src in ("truecar", "craigslist"):
            entry[f"{src}_left"] = (df["source"] == src).sum()
        if step != "raw rows":
            for src in ("truecar", "craigslist"):
                entry[f"removed_{src}"] = (removed["source"] == src).sum()
        log.append(entry)

    drop("raw rows", np.zeros(len(df), dtype=bool))

    # Step 2: missing important features
    for col in IMPORTANT:
        drop(f"missing {col}", df[col].isna())

    # Step 3: bad values
    drop("price < $1,500 (placeholder or down payment)", df["price"] < 1500)
    drop("price > $100,000 (typo)", df["price"] > 100_000)
    drop("mileage > 400,000", df["mileage"] > 400_000)
    drop("mileage < 100 on a 2+ year-old car (placeholder or shorthand)", (df["mileage"] < 100) & (age_of(df) >= 2))
    drop("over 60,000 miles per year", df["mileage"] / age_of(df) > 60_000)
    drop("listing is a different vehicle", df.apply(wrong_vehicle, axis=1))
    drop("price far from similar cars (3x IQR of log-price residual)", price_outliers(df))

    # Step 4: duplicates. An exact odometer reading identifies a car; a round one ("120,000")
    # only does with the same price.
    drop("duplicate VIN (within or across sources)", duplicates(df, ["vin"]))
    drop("duplicate make/model/year/exact mileage (no-VIN cross-posts)",
         duplicates(df, ["make", "model", "year", "mileage"], (df["mileage"] >= 1000) & (df["mileage"] % 1000 != 0)))
    drop("duplicate make/model/year/round mileage/price", duplicates(df, ["make", "model", "year", "mileage", "price"]))
    df = df.assign(_desc=df["description"].str[:300])
    drop("Craigslist repost (same model, price, and description)",
         duplicates(df, ["model", "price", "_desc"], df["source"] == "craigslist"))
    return df, pd.DataFrame(log)


# ---------------------------------------------------------------- step 5: MSRP + statistics

def match_trim_msrp(row, trims_by_ymm):
    """MSRP of the KBB trim whose name starts with the listing's trim ('LE' -> 'LE Sedan 4D').
    Several body styles can share a trim name (F-150 XLT cabs); take their median."""
    trim = row["trim"].lower()
    prices = [m for name, m in trims_by_ymm.get((row["make"], row["model"], row["year"]), [])
              if name.lower() == trim or name.lower().startswith(trim + " ")]
    return float(np.median(prices)) if prices and trim != "unknown" else np.nan


def merge_msrp(df):
    """Attach KBB MSRP: base_msrp per make/model/year (cheapest trim) and trim_msrp when the trim matches."""
    msrp = pd.read_csv(RAW_DIR / "msrp.csv")[["make", "model", "year", "base_msrp", "max_msrp"]]
    df = df.merge(msrp, on=["make", "model", "year"], how="left")
    trims = pd.read_csv(RAW_DIR / "msrp_trims.csv")
    trims_by_ymm = {k: list(zip(g["trim"], g["msrp"])) for k, g in trims.groupby(["make", "model", "year"])}
    df["trim_msrp"] = df.apply(match_trim_msrp, axis=1, trims_by_ymm=trims_by_ymm)
    return df


def attribute_stats(df):
    numeric = ["price", "year", "mileage", "base_msrp", "trim_msrp", "cylinders", "owner_count",
               "accident_count", "mpg_combined"]
    numeric_summary = df[numeric].describe().T.round(1)
    numeric_summary["missing_pct"] = (df[numeric].isna().mean() * 100).round(1)

    categorical = pd.concat({c: df[c].value_counts() for c in ["make", "model"] + CATEGORICAL},
                            names=["attribute", "value"]).reset_index()
    categorical["pct"] = (100 * categorical["count"] / len(df)).round(1)
    categorical["n_unique"] = categorical.groupby("attribute")["value"].transform("size")

    missing = df.replace("unknown", np.nan).groupby("source").apply(
        lambda g: (g.isna().mean() * 100).round(1), include_groups=False).T
    by_source = df.groupby("source").agg(
        listings=("price", "size"), median_price=("price", "median"), mean_price=("price", "mean"),
        median_mileage=("mileage", "median"), median_year=("year", "median"),
        min_year=("year", "min"), max_year=("year", "max"), states=("state", "nunique"),
    ).round(0)
    by_model = df.groupby(["make", "model", "source"]).size().unstack(fill_value=0)
    by_model["total"] = by_model.sum(axis=1)
    return {"numeric_summary": numeric_summary, "categorical_summary": categorical,
            "missing_pct_by_source": missing, "summary_by_source": by_source, "counts_by_model": by_model}


def main():
    truecar = pd.read_csv(RAW_DIR / "truecar.csv")
    craigslist = pd.read_csv(RAW_DIR / "craigslist.csv", dtype={"post_id": str, "zip": str})

    # TrueCar's structured trims are the vocabulary for parsing Craigslist's free-text trims.
    trims_by_model = {k: list(v) for k, v in
                      truecar.dropna(subset=["trim"]).groupby(["make", "search_model"])["trim"].unique().items()}

    combined = pd.concat([unify_truecar(truecar), unify_craigslist(craigslist, trims_by_model)], ignore_index=True)
    clean_df, log = clean(harmonize(combined))
    clean_df = clean_df[COLUMNS].sort_values(["make", "model", "year", "source"]).reset_index(drop=True)
    clean_df[["price", "year", "mileage"]] = clean_df[["price", "year", "mileage"]].astype(int)
    clean_df = merge_msrp(clean_df)

    stats_dir = OUT_DIR / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(OUT_DIR / "listings_clean.csv", index=False)
    log.to_csv(OUT_DIR / "cleaning_log.csv", index=False)
    for name, table in attribute_stats(clean_df).items():
        table.to_csv(stats_dir / f"{name}.csv")

    print(log.astype(object).fillna("").to_string(index=False))
    print(f"\nMSRP matched: base {clean_df['base_msrp'].notna().mean():.1%}, "
          f"trim-level {clean_df['trim_msrp'].notna().mean():.1%}")
    print(f"Saved {len(clean_df)} rows x {clean_df.shape[1]} columns to {OUT_DIR / 'listings_clean.csv'}")


if __name__ == "__main__":
    main()
