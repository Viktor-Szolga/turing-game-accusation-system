import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

root_dir = "early_stopping_fixed_epoch_bow"
#root_dir = "early_stopping_fixed_epoch"

stats = pd.read_csv(os.path.join(root_dir, "architecture_loss_summary.csv"))

stats['architecture'] = stats['n_layers'] + " | " + stats['n_neurons']
stats = stats.sort_values(by='mean_loss', ascending=True).reset_index(drop=True)

sns.set(style="whitegrid")

highlight_arch = "2-layer | 48-24"
colors = ['orange' if arch == highlight_arch else 'skyblue' for arch in stats['architecture']]

plt.figure(figsize=(12, 6))

ax = sns.barplot(
    x='architecture',
    y='mean_loss',
    data=stats,
    palette=colors,
    edgecolor='k'
)

for i, row in stats.iterrows():
    ax.errorbar(
        x=i,
        y=row['mean_loss'],
        yerr=1.96 * row['std_loss'] / np.sqrt(row['num_runs']),  # 95% CI
        fmt='none',
        c='black',
        capsize=5,
        lw=1.5
    )


plt.xticks(rotation=45, ha='right')
plt.ylabel("Mean Validation Loss")
plt.title("Mean Loss per Architecture with 95% Confidence Interval (BoW)")
plt.tight_layout()
plt.show()
