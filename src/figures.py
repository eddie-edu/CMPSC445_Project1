"""Part 6: the three required figures, saved to out/figures/ as 150-dpi PNGs.

  retention_dist.png    data characteristics: value retention ratio distribution by source
  correlation_heatmap.png  feature ranking: feature correlations, ordered by correlation with the target
  model_comparison.png  model comparison: test MAE, RMSE, and R² for each model
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.config import OUT_DIR
from src.features import FLAGS, NUMERIC, TARGET
from src.train import RESULTS_DIR

FIG_DIR = OUT_DIR / "figures"
# Categorical colors in fixed order (validated colorblind-safe palette); text stays in ink colors
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Diverging: orange (negative) -> neutral gray -> blue (positive)
DIVERGING = LinearSegmentedColormap.from_list("orange_gray_blue", ["#c4501f", "#f0efec", "#1d5fae"])
MODEL_LABELS = {"linear": "Linear", "knn": "k-NN", "tree": "Decision tree"}

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 9, "legend.frameon": False, "lines.linewidth": 2,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)
    print(f"  {name}")


def retention_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    for source, color in (("truecar", BLUE), ("craigslist", ORANGE)):
        part = df.loc[df["source"] == source, TARGET]
        ax.hist(part, bins=40, histtype="step", color=color, label=f"{source} (n={len(part)}, median {part.median():.0%})")
    ax.set(title="Value retention ratio (listing price / original MSRP)", xlabel="Retention ratio",
           ylabel="Listings")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend()
    save(fig, "retention_dist.png")


def correlation_heatmap(df):
    """Spearman correlation of the numeric features and flags with each other and with the target,
    ordered by the strength of their correlation with the retention ratio (the ranking)."""
    corr = df[[TARGET] + NUMERIC + FLAGS].corr(method="spearman")
    order = corr[TARGET].abs().sort_values(ascending=False).index
    corr = corr.loc[order, order]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    image = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set(xticks=range(len(corr)), yticks=range(len(corr)),
           title="Feature correlation (Spearman), ranked by correlation with retention ratio")
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iat[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(value) > 0.6 else INK)
    fig.colorbar(image, ax=ax, shrink=0.8)
    save(fig, "correlation_heatmap.png")


def model_comparison():
    """Predicted vs. actual retention ratio for every test listing, one panel per model.
    Dots on the dashed line are perfect predictions."""
    metrics = pd.read_csv(RESULTS_DIR / "metrics.csv").set_index("model")
    predictions = pd.read_csv(RESULTS_DIR / "test_predictions.csv")
    limits = (0, predictions.max().max() * 1.05)
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4.8), sharex=True, sharey=True)
    for ax, (name, row) in zip(axes, metrics.iterrows()):
        ax.scatter(predictions["actual"], predictions[name], s=10, alpha=0.6, color=BLUE)
        ax.plot(limits, limits, color=MUTED, linestyle="--", linewidth=1)
        ax.set(title=f"{MODEL_LABELS[name]}\nMAE {row['test_mae']:.3f}, R² {row['test_r2']:.3f}",
               xlabel="Actual retention ratio", xlim=limits, ylim=limits)
        ax.set_aspect("equal")
    axes[0].set_ylabel(f"Predicted retention ratio ({len(predictions)} test listings)")
    save(fig, "model_comparison.png")


def main():
    FIG_DIR.mkdir(exist_ok=True)
    for old in FIG_DIR.glob("*.png"):
        old.unlink()
    df = pd.read_csv(OUT_DIR / "features.csv")
    retention_distribution(df)
    correlation_heatmap(df)
    model_comparison()


if __name__ == "__main__":
    main()
