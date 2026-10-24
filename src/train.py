"""Part 4: train and compare three regressors that predict the value retention ratio (price / MSRP).

Every model is a Pipeline: impute + scale the numerics, one-hot the low-cardinality categoricals,
and pass the 0/1 flags and binary-encoded trim/state bits through. Each model is tuned with
GridSearchCV (5-fold) on 80% of the rows and scored once on the held-out 20%.

Writes to out/results/:
  metrics.csv           test MAE, RMSE, and R² for each model
  test_predictions.csv  each model's prediction for every test listing
  feature_ranking.csv   the tuned tree's feature importances (the depreciation analysis)
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.config import OUT_DIR
from src.features import NUMERIC, ONE_HOT, TARGET, design_matrix

SEED = 42
RESULTS_DIR = OUT_DIR / "results"

# name -> (estimator, hyperparameter grid)
MODELS = {
    "linear": (LinearRegression(), {}),
    "knn": (KNeighborsRegressor(), {"n_neighbors": [3, 5, 10, 20], "weights": ["uniform", "distance"]}),
    "tree": (DecisionTreeRegressor(random_state=SEED), {"max_depth": [4, 6, 8, 12], "min_samples_leaf": [1, 5, 10]}),
}


def preprocessor():
    """Numerics: median-impute and standardize. Low-cardinality categoricals: one-hot, with categories
    under 10 rows merged. Everything else (0/1 flags, trim/state bits) passes through unchanged."""
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False), ONE_HOT),
    ], remainder="passthrough")


def scores(y_true, y_pred):
    return {"mae": mean_absolute_error(y_true, y_pred), "rmse": root_mean_squared_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)}


def original_feature(name):
    """num__mileage_10k -> mileage_10k, cat__make_Toyota -> make, remainder__trim_b3 -> trim."""
    kind, col = name.split("__", 1)
    if kind == "cat":
        return next(c for c in sorted(ONE_HOT, key=len, reverse=True) if col.startswith(c + "_"))
    return col.rsplit("_b", 1)[0] if col.startswith(("trim_b", "state_b")) else col


def tree_ranking(pipeline):
    """Tree importances summed back to each input feature (one-hot and binary columns -> their source)."""
    names = pipeline["prep"].get_feature_names_out()
    importance = pd.Series(pipeline["reg"].feature_importances_, index=[original_feature(n) for n in names])
    return importance.groupby(level=0).sum().sort_values(ascending=False).rename("importance")


def main():
    df = pd.read_csv(OUT_DIR / "features.csv")
    X, y = design_matrix(df), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=df["model"],
                                                        random_state=SEED)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    rows = []
    predictions = pd.DataFrame({"actual": y_test})
    for name, (estimator, grid) in MODELS.items():
        search = GridSearchCV(Pipeline([("prep", preprocessor()), ("reg", estimator)]),
                              {f"reg__{k}": v for k, v in grid.items()}, cv=cv,
                              scoring="neg_root_mean_squared_error", n_jobs=-1).fit(X_train, y_train)
        predictions[name] = search.predict(X_test)
        test = scores(y_test, predictions[name])
        rows.append({"model": name, "cv_rmse": -search.best_score_, **{f"test_{k}": v for k, v in test.items()},
                     "best_params": {k.removeprefix("reg__"): v for k, v in search.best_params_.items()}})
        if name == "tree":
            ranking = tree_ranking(search.best_estimator_)
            ranking.to_csv(RESULTS_DIR / "feature_ranking.csv", index_label="feature")

    metrics = pd.DataFrame(rows)
    metrics.round(4).to_csv(RESULTS_DIR / "metrics.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    print(f"train {len(X_train)} rows, test {len(X_test)} rows")
    print(metrics.round(4).to_string(index=False))
    print("\nDepreciation analysis - feature ranking (decision tree importance):")
    print(ranking.round(3).to_string())


if __name__ == "__main__":
    main()
