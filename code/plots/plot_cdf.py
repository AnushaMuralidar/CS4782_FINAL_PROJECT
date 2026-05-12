import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

RESULTS_DIR = "results"
FIGURE_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

cases = [
    ("baseline_s1/baseline_s1_final_errors.npy", "Baseline (ResNet)", 0),
    ("unet_s1/unet_s1_final_errors.npy", "UNet", 0),
    ("transformer_s1/transformer_s1_final_errors.npy", "UNet + Transformer", 0),
    ("crossap_s1/crossap_s1_final_errors.npy", "Cross-AP Transformer", 0),

    ("baseline_s2/baseline_s2_final_errors.npy", "Baseline (ResNet)", 1),
    ("unet_s2/unet_s2_final_errors.npy", "UNet", 1),
    ("transformer_s2/transformer_s2_final_errors.npy", "UNet + Transformer", 1),
    ("crossap_s2/crossap_s2_final_errors.npy", "Cross-AP Transformer", 1),
]

STYLE_MAP = {
    "Baseline (ResNet)": "--",
    "UNet": "-",
    "UNet + Transformer": ":",
    "Cross-AP Transformer": "-.",
}

COLOR_MAP = {
    "Baseline (ResNet)": "tab:red",
    "UNet": "tab:green",
    "UNet + Transformer": "tab:blue",
    "Cross-AP Transformer": "tab:purple",
}


def smooth_empirical_cdf(errors_m, x_max_m, n_points=2000, bw=0.08):
    reflected = np.concatenate([-errors_m, errors_m])
    kde = gaussian_kde(reflected, bw_method=bw)

    x = np.linspace(0, x_max_m, n_points)
    pdf = kde(x) * 2
    pdf = np.clip(pdf, 0, None)

    cdf = np.cumsum(pdf) * (x[1] - x[0])
    cdf = cdf / cdf[-1]

    return x, cdf


def add_cdf(ax, errors_m, label, x_max_m):
    med_cm = np.median(errors_m) * 100
    p90_cm = np.percentile(errors_m, 90) * 100

    x, y = smooth_empirical_cdf(errors_m, x_max_m)

    ax.plot(
        x,
        y,
        linestyle=STYLE_MAP[label],
        linewidth=2.5,
        color=COLOR_MAP[label],
        label=f"{label} | med={med_cm:.0f} cm, p90={p90_cm:.0f} cm",
    )


fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
group_titles = ["Single Scenario", "Multi Scenario"]

for group_idx in range(2):
    ax = axes[group_idx]
    group_cases = [c for c in cases if c[2] == group_idx]

    all_errors = {}

    for rel_path, label, _ in group_cases:
        full_path = os.path.join(RESULTS_DIR, rel_path)

        if not os.path.exists(full_path):
            print(f"Missing file: {full_path}")
            continue

        e = np.load(full_path)

        # If errors look like cm, convert to meters
        if np.median(e) > 5:
            e = e / 100.0

        all_errors[label] = e

    x_max_m = max(np.percentile(e, 90) for e in all_errors.values()) * 2.5

    for label, errors in all_errors.items():
        add_cdf(ax, errors, label, x_max_m)

    for yval, ytxt in [(0.5, "50%"), (0.9, "90%")]:
        ax.axhline(yval, color="black", linestyle=":", linewidth=1.2)
        ax.text(
            0.01,
            yval + 0.01,
            ytxt,
            transform=ax.get_yaxis_transform(),
            fontsize=9,
            va="bottom",
        )

    ax.set_title(group_titles[group_idx], fontsize=20, fontweight="bold")
    ax.set_xlabel("Localization Error (m)", fontsize=18, fontweight="bold")
    ax.set_ylabel("CDF", fontsize=18, fontweight="bold")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, x_max_m)
    ax.tick_params(axis="both", labelsize=16, width=1.5, length=6)
    ax.legend(fontsize=11, loc="lower right", framealpha=0.9, edgecolor="black")
    ax.grid(True, alpha=0.4, linewidth=1.2)

plt.tight_layout()

out = os.path.join(FIGURE_DIR, "cdf_plot.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {out}")