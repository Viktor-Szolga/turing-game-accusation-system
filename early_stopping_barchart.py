import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load CSV
stats = pd.read_csv("architecture_loss_summary.csv")

# Create a simple string for architecture
stats['architecture'] = stats['n_layers'] + " | " + stats['n_neurons']
stats = stats.sort_values(by='mean_loss', ascending=True).reset_index(drop=True)

# Set style
sns.set(style="whitegrid")

# Create color palette: highlight 48-22 architecture
highlight_arch = "2-layer | 48-24 neurons"
colors = ['orange' if arch == highlight_arch else 'skyblue' for arch in stats['architecture']]

# Create figure
plt.figure(figsize=(12, 6))

# Plot mean values as bars with custom colors
ax = sns.barplot(
    x='architecture',
    y='mean_loss',
    data=stats,
    palette=colors,
    edgecolor='k'
)

# Add 95% CI error bars manually
for i, row in stats.iterrows():
    ax.errorbar(
        x=i,
        y=row['mean_loss'],
        yerr=1.96 * row['std_loss'] / np.sqrt(row['num_runs']),  # 95% CI
        fmt='none',  # no marker
        c='black',
        capsize=5,
        lw=1.5
    )

# Labels and layout
plt.xticks(rotation=45, ha='right')
plt.ylabel("Mean Validation Loss")
plt.title("Mean Loss per Architecture with 95% Confidence Interval")
plt.tight_layout()
plt.show()
