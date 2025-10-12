import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from itertools import cycle
import matplotlib.ticker as ticker

root_dir = "gridsearch"
run_names = [
    "run072",
    "run077",
    "run082",
    "run294",
    "run295",
]

legends = [
    "B | lr=0.0001 | Dropout=0.85 | Weight decay=0.0",
    "B | lr=0.0005 | Dropout=0.85 | Weight decay=0.0",
    "B | lr=0.001 | Dropout=0.85 | Weight decay=0.0",
    "B | lr=0.01 | Dropout=0.85 | Weight decay=0.0",
    "B | lr=0.05 | Dropout=0.85 | Weight decay=0.0",
]
model_regularizations = [
    "l",
    "l",
    "m",
    "m",
    "h"
]

print("Plotting runs:")
for r in run_names:
    print(r)

val_losses = []

for run in run_names:
    with open(os.path.join(root_dir, "training_information", f"{run}.pkl"), "rb") as f:
        data = pickle.load(f)
        val_losses.append(data["validation_loss"])


reg_technique_colors = {"l": "tab:blue", "m": "tab:orange", "h": "tab:green", "Linear": "tab:purple"}


line_styles = ["-", ":", "--", "-", ":", "--", "-"]
line_widths = [1, 2, 1.5, 1, 2, 1.5, 1]
alphas = cycle([0.7])

fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("Investigating different learning rates", y=0.94)


for validation_loss, model, legend, ls, lw, alpha in zip(
    val_losses, model_regularizations, legends, line_styles, line_widths, alphas
):
    ax.plot(validation_loss, label=legend, color=reg_technique_colors[model],
            linestyle=ls, lw=lw, alpha=alpha)

ax.set_yscale("log")
ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())

ax.set_xlabel("Epochs")
ax.set_ylabel("Validation loss (log scale)")
ax.grid(True, alpha=0.3)


plt.tight_layout(rect=[0, 0.15, 1, 0.95])
fig.legend(legends, loc="lower center", ncol=2, frameon=False)

plt.show()
