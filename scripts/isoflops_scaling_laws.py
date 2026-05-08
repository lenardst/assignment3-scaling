import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

runs = json.load(open("data/isoflops_curves.json"))

compute_budgets = np.array(sorted(set(run["compute_budget"] for run in runs)))

optimal_runs = [
    min(
        (run for run in runs if run["compute_budget"] == budget),
        key=lambda run: run["final_loss"],
    )
    for budget in compute_budgets
]

optimal_model_sizes = np.array([run["parameters"] for run in optimal_runs])
optimal_dataset_sizes = np.array([
    run["compute_budget"] / (6 * run["parameters"]) for run in optimal_runs
])


def log_power_law(log_compute, log_scale, exponent):
    return log_scale + exponent * log_compute


model_size_fit, _ = curve_fit(log_power_law, np.log(compute_budgets), np.log(optimal_model_sizes))
dataset_size_fit, _ = curve_fit(log_power_law, np.log(compute_budgets), np.log(optimal_dataset_sizes))


def predict_model_size(compute):
    return np.exp(log_power_law(np.log(compute), *model_size_fit))


def predict_dataset_size(compute):
    return np.exp(log_power_law(np.log(compute), *dataset_size_fit))


extrapolation_budgets = np.logspace(np.log10(compute_budgets.min()), 24, 500)

for target_budget in [1e23, 1e24]:
    print(
        f"Predicted optimal model size at C={target_budget:.0e}: "
        f"{predict_model_size(target_budget):.3e} parameters"
    )
    print(
        f"Predicted optimal dataset size at C={target_budget:.0e}: "
        f"{predict_dataset_size(target_budget):.3e} tokens"
    )


model_size_exponent = model_size_fit[1]
dataset_size_exponent = dataset_size_fit[1]

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(compute_budgets, optimal_model_sizes, color="steelblue", zorder=5,
           label=r"$N_\mathrm{opt}(C_i)$ from IsoFLOPs profiles")
ax.plot(extrapolation_budgets, predict_model_size(extrapolation_budgets), color="steelblue",
        label=rf"Power law fit ($N_\mathrm{{opt}} \propto C^{{{model_size_exponent:.3f}}}$)")
for budget, label in [(1e23, r"$10^{23}$"), (1e24, r"$10^{24}$")]:
    ax.axvline(budget, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.annotate(label, xy=(budget, ax.get_ylim()[0]), xytext=(budget * 1.1, predict_model_size(budget) * 0.5),
                fontsize=9, color="gray")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Compute budget $C$ (FLOPs)")
ax.set_ylabel(r"Optimal model size $N_\mathrm{opt}$ (parameters)")
ax.set_title("IsoFLOPs: Compute-Optimal Model Size")
ax.legend()
plt.tight_layout()
plt.savefig("figures/model_size_scaling.pdf", bbox_inches="tight")
plt.savefig("figures/model_size_scaling.png", dpi=150, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(compute_budgets, optimal_dataset_sizes, color="darkorange", zorder=5,
           label=r"$D_\mathrm{opt}(C_i)$ from IsoFLOPs profiles")
ax.plot(extrapolation_budgets, predict_dataset_size(extrapolation_budgets), color="darkorange",
        label=rf"Power law fit ($D_\mathrm{{opt}} \propto C^{{{dataset_size_exponent:.3f}}}$)")
for budget, label in [(1e23, r"$10^{23}$"), (1e24, r"$10^{24}$")]:
    ax.axvline(budget, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.annotate(label, xy=(budget, ax.get_ylim()[0]), xytext=(budget * 1.1, predict_dataset_size(budget) * 0.5),
                fontsize=9, color="gray")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Compute budget $C$ (FLOPs)")
ax.set_ylabel(r"Optimal dataset size $D_\mathrm{opt}$ (tokens)")
ax.set_title("IsoFLOPs: Compute-Optimal Dataset Size")
ax.legend()
plt.tight_layout()
plt.savefig("figures/dataset_size_scaling.pdf", bbox_inches="tight")
plt.savefig("figures/dataset_size_scaling.png", dpi=150, bbox_inches="tight")
plt.show()
