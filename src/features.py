"""Part 3: feature engineering on the cleaned listings.

Adds the target (value retention ratio = price / base MSRP), numeric features (age, mileage, MSRP),
and 0/1 flags parsed from the Craigslist description, then writes out/features.csv.
design_matrix() picks the model inputs and binary-encodes the high-cardinality columns (trim, state).
"""
import numpy as np
import pandas as pd

from src.config import OUT_DIR
from src.preprocess import REFERENCE_YEAR

TARGET = "retention_ratio"

# flag -> phrases searched in the Craigslist description (lower-cased, dashes replaced with spaces)
TEXT_FLAGS = {
    "one_owner": r"\b(?:one owner|1 owner|single owner)\b",
    "no_accidents": r"\b(?:no accidents|accident free|clean history)\b",
    "needs_work": r"\b(?:needs work|as is|check engine|won['’]t start|will not start)\b",
}

# Model inputs, grouped by how the pipeline treats them
# Only fields both sources provide, and no `source` column: every listing is treated the same way
NUMERIC = ["vehicle_age", "mileage_10k", "miles_per_year", "log_base_msrp", "cylinders"]
FLAGS = list(TEXT_FLAGS)
ONE_HOT = ["make", "model", "body_type", "drivetrain", "transmission", "fuel_type", "title_status",
           "seller_type"]
HIGH_CARDINALITY = ["trim", "state"]


def add_features(df):
    df = df.copy()
    df[TARGET] = df["price"] / df["base_msrp"]
    df["vehicle_age"] = REFERENCE_YEAR - df["year"]
    df["mileage_10k"] = df["mileage"] / 10_000
    df["miles_per_year"] = df["mileage"] / df["vehicle_age"]
    df["log_base_msrp"] = np.log(df["base_msrp"])

    # Sellers write "1-owner" and "1 owner" alike, so dashes become spaces before matching
    text = df["description"].fillna("").str.lower().str.replace("-", " ")
    for flag, pattern in TEXT_FLAGS.items():
        df[flag] = text.str.contains(pattern, regex=True).astype(int)
    return df


def binary_encode(df, columns):
    """Replace each column with ceil(log2(n_categories)) 0/1 columns holding its category code in binary."""
    df = df.copy()
    for col in columns:
        codes = df[col].astype("category").cat.codes
        for i in range(max(int(codes.max()).bit_length(), 1)):
            df[f"{col}_b{i}"] = codes // 2**i % 2
    return df.drop(columns=columns)


def design_matrix(df):
    """Model inputs X, with trim (86 values -> 7 bits) and state (26 -> 5 bits) binary-encoded."""
    return binary_encode(df[NUMERIC + FLAGS + ONE_HOT + HIGH_CARDINALITY], HIGH_CARDINALITY)


def main():
    df = add_features(pd.read_csv(OUT_DIR / "listings_clean.csv"))
    df.to_csv(OUT_DIR / "features.csv", index=False)
    print(df.groupby("source")[FLAGS].mean().mul(100).round(1).to_string())
    print(f"Saved {len(df)} rows x {df.shape[1]} columns to {OUT_DIR / 'features.csv'}")


if __name__ == "__main__":
    main()
