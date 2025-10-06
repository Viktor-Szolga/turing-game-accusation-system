from src.utils import plot_curves_on_ax
import pickle
import os
import matplotlib.pyplot as plt
from itertools import cycle
from extract_runs import load_runs, query_runs
import matplotlib.ticker as ticker


runs = load_runs("runs.txt")
# Example query
res = query_runs(runs, model="Model B", dropout=0.85, weight_decay=0)
#res=res[-9:] # use for dropout 0.0 and weight_decay 0.0 to filter duplicates

run_names = [r["run_id"] for r in res]
legends = [f'{r["model"].split()[-1]} | lr={r["lr"]} | Dropout={r["dropout"]} | Weight decay={r["weight_decay"]}' for r in res]
models = [r["model"] for r in res]
models = [
    "Model A",
    "Model A",
    "Model B",
    "Model B",
    "Model C",
    "Model C",
    ]
print(f"Plotting runs:")
for r in run_names:
    print(r)

train_losses, val_losses = [], []


for run in run_names:
    with open(os.path.join("training_information", f"{run}.pkl"), "rb") as f:
        data = pickle.load(f)
        train_losses.append(data["train_loss"])
        val_losses.append(data["validation_loss"])


model_colors = {"Model A": "tab:blue", "Model B": "tab:orange", "Model C": "tab:green"}


line_styles = cycle(["-", "--", ":", ]) #"-."
line_widths = cycle([1, 1.5, 1.5,]) # 2
alphas = cycle([0.4, 0.6, 0.8, 1.0])

fig, axs = plt.subplots(1, 2, figsize=(10, 4))


fig.suptitle("Model B with dropout .85 and varying learning rates", y=0.82)


for train_loss, validation_loss, model, legend, ls, lw, alpha in zip(
    train_losses, val_losses, models, legends, line_styles, line_widths, alphas
):
    axs[0].plot(train_loss, label=legend, color=model_colors[model], linestyle=ls, lw=lw, alpha=alpha)
    axs[1].plot(validation_loss, label=legend, color=model_colors[model], linestyle=ls, lw=lw, alpha=alpha)

axs[0].set_title("Train Loss")
axs[0].set_yscale("log")
axs[0].set_ylabel("Loss (log scale)")

axs[1].set_title("Validation Loss")
axs[1].set_yscale("log")
axs[0].yaxis.set_minor_formatter(ticker.ScalarFormatter())
axs[1].yaxis.set_minor_formatter(ticker.ScalarFormatter())


plt.tight_layout(rect=[0, 0.2, 1, 0.9]) 


fig.legend(legends, loc="lower center", ncol=3, frameon=False)

plt.show()
