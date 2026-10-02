"""Supplementary model-comparison view; full-range registered figures stay intact."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase_c"
frame = pd.read_csv(OUT / "tables/model_horizon_pooled_metrics.csv")
frame = frame.loc[frame.dataset.eq("D1")]
styles = {
    "B1": ("#777777", "--", "Weekly baseline"),
    "B5": ("#1668ad", "-", "Kalman: selected by complexity tie"),
    "M1": ("#cb7300", "-", "LightGBM"),
    "M1-W": ("#2c9463", "-", "Peak-weighted LightGBM"),
    "C1": ("#9865a5", "-", "Residual LightGBM"),
    "M2": ("#ce493f", "-", "TCN: fails peak safeguard"),
    "R1": ("#222222", ":", "Chronos-2: reference only"),
}
fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
for ax, metric, label in zip(axes,("MAE","Peak_MAE"),("MAE","Peak MAE")):
    for model, (color, linestyle, description) in styles.items():
        cell = frame.loc[frame.model.eq(model)].sort_values("horizon")
        ax.plot(cell.horizon * 15, cell[metric], color=color, linestyle=linestyle,
                linewidth=2.8 if model == "B5" else 1.5, marker="o", markersize=3,
                label=f"{model} · {description}")
    ax.set(xlabel="Forecast horizon (minutes)", ylabel=label, xticks=[60,90,120,150,180,210,240])
    ax.grid(alpha=.2)
    ax.spines[["top","right"]].set_visible(False)
axes[0].set_title("Average error across shared development cases",fontsize=11)
axes[1].set_title("Error on actual fit-Q95 exceedances",fontsize=11)
fig.suptitle("Phase C · Selected family, comparators and reference",fontsize=14)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc="outside lower center",ncol=2,fontsize=8,frameon=False)
path = OUT / "figures/selection_detail.png"
fig.savefig(path,dpi=180,metadata={"Description":"Development-only supplementary subset. Full all-model figures retained. No horizon selected."})
plt.close(fig)
print(path)
