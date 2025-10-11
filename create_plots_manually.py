import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from itertools import cycle
import matplotlib.ticker as ticker

run_names = [
    "run000",
    #"run293"
]

legends = [
    "B | lr=0.001 | Dropout=0.85 | Weight decay=0",
    #"Linear | lr=0.001 | Dropout=0 | Weight decay=0"
]

model_regularizations = [
    "B",
    #"Linear"
]


print("Plotting runs:")
for r in run_names:
    print(r)

val_losses = []

for run in run_names:
    with open(os.path.join("training_information", f"{run}.pkl"), "rb") as f:
        data = pickle.load(f)
        val_losses.append(data["validation_loss"])


reg_technique_colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "Linear": "tab:purple"}


line_styles = cycle(["-", "--", ":", "-."])
line_widths = cycle([1, 1.5, 1.5])
alphas = cycle([0.7])

line_styles = ["-", "-."]


fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("Linear classifier compared to the best run", y=0.94)


for validation_loss, model, legend, ls, lw, alpha in zip(
    val_losses, model_regularizations, legends, line_styles, line_widths, alphas
):
    ax.plot(validation_loss, label=legend, color=reg_technique_colors[model],
            linestyle=ls, lw=lw, alpha=alpha)

ax.set_title("Linear classifier compared to the best run")
ax.set_yscale("log")
ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())

ax.set_xlabel("Epochs")
ax.set_ylabel("Validation loss (log scale)")
ax.grid(True, alpha=0.3)


plt.tight_layout(rect=[0, 0.15, 1, 0.95])
fig.legend(legends, loc="lower center", ncol=2, frameon=False)

plt.show()
