import os
import numpy as np
import matplotlib.pyplot as plt

LAMBDA_DIR = "results/lambda_sweep"
FIGURE_DIR = "results/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

cases = [
    ("lambda_0_final_errors.npy", "0.0"),
    ("lambda_001_final_errors.npy", "0.01"),
    ("lambda_01_final_errors.npy", "0.1"),
    ("lambda_05_final_errors.npy", "0.5"),
]

x_labels = []
median_errors = []
p90_errors = []

for fname, label in cases:
    path = os.path.join(LAMBDA_DIR, fname)
    errors_m = np.load(path)  # errors are stored in meters

    med_cm = np.median(errors_m) * 100
    p90_cm = np.percentile(errors_m, 90) * 100

    x_labels.append(label)
    median_errors.append(med_cm)
    p90_errors.append(p90_cm)

x = np.arange(len(x_labels))

plt.figure(figsize=(8.5, 5.5))

plt.plot(x, median_errors, marker="o", markersize=12, linewidth=2.8, label="Median Error")
plt.plot(x, p90_errors, marker="s", markersize=12, linewidth=2.8, linestyle="--", label="90th Percentile Error", color = 'red')

for xi, y in zip(x, median_errors):
    plt.text(xi, y + 35, f"{y:.0f} cm", ha="center", fontsize=11)

for xi, y in zip(x, p90_errors):
    plt.text(xi, y + 35, f"{y:.0f} cm", ha="center", fontsize=11)

plt.title("Localization Error vs. Consistency Loss Weight", fontsize=20)
plt.xlabel(r"Consistency Loss Weight ($\lambda_c$)", fontsize=18)
plt.ylabel("Localization Error (cm)", fontsize=18)

plt.xticks(x, x_labels, fontsize=14)
plt.yticks(fontsize=14)

plt.grid(True, alpha=0.3)
plt.legend(fontsize=14)
plt.tight_layout()

save_path = os.path.join(FIGURE_DIR, "lambda_sweep_tradeoff.png")
plt.savefig(save_path, dpi=300)
plt.show()
