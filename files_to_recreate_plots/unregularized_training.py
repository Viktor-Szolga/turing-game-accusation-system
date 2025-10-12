import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import plot_curves_on_ax
import pickle
import matplotlib.pyplot as plt
from itertools import cycle
from extract_runs import load_runs, query_runs
import matplotlib.ticker as ticker

root_dir = os.path.join("gridsearch")
runs = load_runs(os.path.join(root_dir, "runs.txt"))
# Example query
res = query_runs(runs, model="Model B", dropout=0.85, weight_decay=0)
res = query_runs(runs, dropout=0, weight_decay=0)
res=res[-9:] # use for dropout 0.0 and weight_decay 0.0 to filter duplicates

run_names = [r["run_id"] for r in res]
legends = [f'{r["model"].split()[-1]} | lr={r["lr"]} | Dropout={r["dropout"]} | Weight decay={r["weight_decay"]}' for r in res]
models = [r["model"] for r in res]

print(f"Plotting runs:")
for r in run_names:
    print(r)

train_losses, val_losses = [], []


for run in run_names:
    with open(os.path.join(root_dir, "training_information", f"{run}.pkl"), "rb") as f:
        data = pickle.load(f)
        train_losses.append(data["train_loss"])
        val_losses.append(data["validation_loss"])


model_colors = {"Model A": "tab:blue", "Model B": "tab:orange", "Model C": "tab:green"}


line_styles = cycle(["-", "--", ":", ]) #"-."
line_widths = cycle([1, 1.5, 1.5,]) # 2
alphas = cycle([0.4, 0.6, 0.8, 1.0])
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=False, gridspec_kw={'wspace': 0.25})

fig.subplots_adjust(top=0.82, bottom=0.25, left=0.08, right=0.98, wspace=0.25)

fig.suptitle("Models A, B and C without regularization", y=0.97, fontsize=12)

for train_loss, validation_loss, model, legend, ls, lw, alpha in zip(
    train_losses, val_losses, models, legends, line_styles, line_widths, alphas
):
    axs[0].plot(train_loss, label=legend, color=model_colors[model], linestyle=ls, lw=lw, alpha=alpha)
    axs[1].plot(validation_loss, label=legend, color=model_colors[model], linestyle=ls, lw=lw, alpha=alpha)

axs[0].set_title("Train Loss")
axs[1].set_title("Validation Loss")

for ax in axs:
    ax.set_yscale("log")
    ax.set_ylabel("Loss (log scale)")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())

fig.legend(legends, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.02))

plt.show()