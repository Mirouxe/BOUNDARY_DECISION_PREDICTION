"""
Boundary Decision Prediction — Circular boundary classification.

Classifies data points (I, F) into 4 classes (E) separated by
quarter-circle boundaries in the first quadrant (I >= 0, F >= 0).

The optimal radii are found by minimizing the classification error
via scipy.optimize, then displayed on a publication-quality 2D plot.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from scipy.optimize import minimize
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Synthetic dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(n_total: int = 54, seed: int = 42) -> pd.DataFrame:
    """
    Generate *n_total* points in the first quadrant (I >= 0, F >= 0)
    with 4 classes (E = 1, 2, 3, 4) separated by three concentric
    quarter-circles of true radii r1 < r2 < r3.
    """
    rng = np.random.default_rng(seed)

    true_radii = [3.0, 6.0, 9.0]
    n_per_class = n_total // 4
    remainder = n_total - 4 * n_per_class

    records = []
    bounds = [0.0] + true_radii + [13.0]

    for cls_idx in range(4):
        r_lo, r_hi = bounds[cls_idx], bounds[cls_idx + 1]
        n_cls = n_per_class + (1 if cls_idx < remainder else 0)

        r = rng.uniform(r_lo + 0.3, r_hi - 0.3, size=n_cls)
        theta = rng.uniform(0, np.pi / 2, size=n_cls)

        I_vals = r * np.cos(theta)
        F_vals = r * np.sin(theta)

        for i in range(n_cls):
            records.append({"I": I_vals[i], "F": F_vals[i], "E": cls_idx + 1})

    df = pd.DataFrame(records)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Optimal circle-boundary fitting
# ---------------------------------------------------------------------------

def classify_by_radii(I: np.ndarray, F: np.ndarray,
                      radii: np.ndarray) -> np.ndarray:
    """Assign class labels 1..4 based on distance to origin and 3 radii."""
    dist = np.sqrt(I**2 + F**2)
    labels = np.ones(len(dist), dtype=int)
    for k, r in enumerate(sorted(radii)):
        labels[dist > r] = k + 2
    return labels


def fit_circle_boundaries(df: pd.DataFrame, n_classes: int = 4):
    """
    Find the 3 radii (r1 < r2 < r3) that best separate the 4 classes
    by minimising the misclassification rate.
    """
    I = df["I"].values
    F = df["F"].values
    y_true = df["E"].values

    def objective(params):
        radii = np.sort(params)
        y_pred = classify_by_radii(I, F, radii)
        error = np.mean(y_pred != y_true)
        penalty = 1e3 * np.sum(np.maximum(0, -np.diff(radii)))
        return error + penalty

    best_result = None
    for r1_init in np.linspace(1, 5, 5):
        for r2_init in np.linspace(4, 8, 5):
            for r3_init in np.linspace(7, 12, 5):
                if r1_init >= r2_init or r2_init >= r3_init:
                    continue
                x0 = [r1_init, r2_init, r3_init]
                res = minimize(objective, x0, method="Nelder-Mead",
                               options={"maxiter": 5000, "xatol": 1e-6})
                if best_result is None or res.fun < best_result.fun:
                    best_result = res

    optimal_radii = np.sort(best_result.x)
    return optimal_radii, best_result.fun


# ---------------------------------------------------------------------------
# 3. Publication-quality plot
# ---------------------------------------------------------------------------

CLASS_LABELS = {1: "E = 1", 2: "E = 2", 3: "E = 3", 4: "E = 4"}
CLASS_COLORS = {1: "#1b9e77", 2: "#d95f02", 3: "#7570b3", 4: "#e7298a"}
CLASS_MARKERS = {1: "o", 2: "s", 3: "^", 4: "D"}


def plot_boundaries(df: pd.DataFrame, radii: np.ndarray,
                    accuracy: float, save_path: str = "boundary_plot.png"):
    """Create a scientific-quality 2D scatter plot with circle boundaries."""

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "figure.dpi": 150,
    })

    fig, ax = plt.subplots(figsize=(7, 6.5))

    for cls in sorted(df["E"].unique()):
        subset = df[df["E"] == cls]
        ax.scatter(subset["I"], subset["F"],
                   c=CLASS_COLORS[cls],
                   marker=CLASS_MARKERS[cls],
                   s=60, edgecolors="k", linewidths=0.5,
                   label=CLASS_LABELS[cls], zorder=5)

    theta_fill = np.linspace(0, np.pi / 2, 300)
    all_bounds = [0.0] + list(radii) + [max(df["I"].max(), df["F"].max()) + 2]
    for i in range(4):
        r_inner = all_bounds[i]
        r_outer = all_bounds[i + 1]
        x_outer = r_outer * np.cos(theta_fill)
        y_outer = r_outer * np.sin(theta_fill)
        x_inner = r_inner * np.cos(theta_fill[::-1])
        y_inner = r_inner * np.sin(theta_fill[::-1])
        xs = np.concatenate([x_outer, x_inner])
        ys = np.concatenate([y_outer, y_inner])
        ax.fill(xs, ys, color=CLASS_COLORS[i + 1], alpha=0.10, zorder=1)

    theta_arc = np.linspace(0, np.pi / 2, 300)
    for k, r in enumerate(radii):
        x_arc = r * np.cos(theta_arc)
        y_arc = r * np.sin(theta_arc)
        ax.plot(x_arc, y_arc, color="k", linewidth=1.2,
                linestyle="--", zorder=4)
        angle_label = np.pi / 4
        ax.annotate(
            f"$r_{k+1} = {r:.2f}$",
            xy=(r * np.cos(angle_label), r * np.sin(angle_label)),
            xytext=(12, 12), textcoords="offset points",
            fontsize=9, fontstyle="italic",
            arrowprops=dict(arrowstyle="->", color="grey", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="grey", alpha=0.9),
            zorder=6)

    ax.set_xlabel("$I$", fontsize=13)
    ax.set_ylabel("$F$", fontsize=13)
    ax.set_title("Boundary Decision Prediction\n"
                 "Quarter-circle classification boundaries",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="grey",
              fontsize=10, title="Classes", title_fontsize=11)

    acc_text = f"Accuracy = {accuracy:.1%}"
    eq_lines = "\n".join(
        [f"$r_{k+1}$: $I^2 + F^2 = {r**2:.2f}$  ($r = {r:.2f}$)"
         for k, r in enumerate(radii)]
    )
    info_text = f"{acc_text}\n\nBoundary equations:\n{eq_lines}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=8.5, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow",
                      ec="grey", alpha=0.9),
            zorder=7)

    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  BOUNDARY DECISION PREDICTION")
    print("  Quarter-circle classification")
    print("=" * 60)

    df = generate_dataset(n_total=54, seed=42)
    csv_path = Path("dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDataset ({len(df)} samples) saved to {csv_path}")
    print(df.groupby("E").size().rename("count").to_frame())

    print("\nFitting optimal circle boundaries ...")
    radii, error = fit_circle_boundaries(df)
    accuracy = 1.0 - error

    print(f"\nOptimal radii found:")
    for k, r in enumerate(radii):
        print(f"  r{k+1} = {r:.4f}  →  I² + F² = {r**2:.4f}")
    print(f"\nClassification accuracy: {accuracy:.1%}")

    y_pred = classify_by_radii(df["I"].values, df["F"].values, radii)
    df["E_pred"] = y_pred
    df.to_csv("dataset_with_predictions.csv", index=False)

    plot_path = plot_boundaries(df, radii, accuracy)
    print(f"\nDone. Results in: {csv_path}, dataset_with_predictions.csv, {plot_path}")


if __name__ == "__main__":
    main()
