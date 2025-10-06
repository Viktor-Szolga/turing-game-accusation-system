import math
import os
import pickle
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D


def load_runs(filepath: str):
    """Load run configurations from file."""
    runs = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            run_id, details = line.split(":", 1)
            parts = [p.strip() for p in details.split("|")]
            run = {"run_id": run_id.replace(".yaml", "").strip()}
            for p in parts:
                if p.startswith("Model"):
                    run["model"] = p
                else:
                    key, val = p.split("=")
                    run[key.strip()] = float(val.strip())
            runs.append(run)
    return runs


def load_validation_data(run_names, data_dir="training_information"):
    """Load validation loss data for given run names."""
    validation_data = {}
    
    for run in run_names:
        try:
            with open(os.path.join(data_dir, f"{run}.pkl"), "rb") as f:
                data = pickle.load(f)
                val_loss = data["validation_loss"][-1] if data["validation_loss"] else float('inf')
                validation_data[run] = val_loss
        except (FileNotFoundError, Exception):
            validation_data[run] = float('inf')
    
    return validation_data


def create_heatmap_data(runs, validation_data):
    """Create data structure for plotting grouped by learning rate and model."""
    allowed_weight_decays = {1e-4, 5e-4, 1e-3, 5e-3}
    
    filtered_runs = [
        run for run in runs 
        if (run['dropout'] >= 0.55 and 
            any(math.isclose(run['weight_decay'], wd, rel_tol=1e-9, abs_tol=1e-12) 
                for wd in allowed_weight_decays))
    ]
    
    lr_model_groups = defaultdict(lambda: defaultdict(list))
    for run in filtered_runs:
        lr_model_groups[run['lr']][run['model']].append(run)
    
    heatmap_data = {}
    
    for lr, model_groups in lr_model_groups.items():
        heatmap_data[lr] = {}
        all_dropouts = set()
        all_weight_decays = set()
        for model_runs in model_groups.values():
            all_dropouts.update([run['dropout'] for run in model_runs])
            all_weight_decays.update([run['weight_decay'] for run in model_runs])
        
        dropouts = sorted([d for d in all_dropouts if d >= 0.55])
        weight_decays = sorted([wd for wd in all_weight_decays 
                               if any(math.isclose(wd, awd, rel_tol=1e-9, abs_tol=1e-12) 
                                     for awd in allowed_weight_decays)])
        
        for model, model_runs in model_groups.items():
            matrix = np.full((len(dropouts), len(weight_decays)), np.nan)
            
            for run in model_runs:
                if run['run_id'] in validation_data:
                    dropout_idx = dropouts.index(run['dropout'])
                    weight_decay_idx = weight_decays.index(run['weight_decay'])
                    matrix[dropout_idx, weight_decay_idx] = validation_data[run['run_id']]
            
            heatmap_data[lr][model] = {
                'matrix': matrix,
                'dropouts': dropouts,
                'weight_decays': weight_decays
            }
    
    return heatmap_data


def plot_meshgrids(heatmap_data, figsize_per_model=(7, 6)):
    """Plot validation loss as 3D meshgrids for each learning rate with models side by side.
    All subplots for a given learning rate share the same x, y, z, and color scales."""
    if not heatmap_data:
        print("No data to plot!")
        return
    
    for lr, lr_data in sorted(heatmap_data.items()):
        if lr not in [0.0001, 0.0005, 0.001]:
            continue
        models = sorted(lr_data.keys())
        if "Model C1000" in models:
            models.remove("Model C1000")
        n_models = len(models)
        
        if n_models == 0:
            continue
        
        all_dropouts = []
        all_weight_decays = []
        all_losses = []
        for model, data in lr_data.items():
            if data['matrix'].size > 0:
                all_dropouts.extend(data['dropouts'])
                all_weight_decays.extend(data['weight_decays'])
                all_losses.extend(data['matrix'][~np.isnan(data['matrix'])].flatten())
        
        if not all_losses:
            print(f"No valid data for learning rate {lr}, skipping.")
            continue
        
        x_min, x_max = min(all_weight_decays), max(all_weight_decays)
        y_min, y_max = min(all_dropouts), max(all_dropouts)
        z_min, z_max = min(all_losses), max(all_losses)

        fig = plt.figure(figsize=(figsize_per_model[0] * n_models + 2, figsize_per_model[1]))  # +2 for space
        axes = []
        surfaces = []

        for i, model in enumerate(models, 1):
            data = lr_data[model]
            dropouts = np.array(data['dropouts'])
            weight_decays = np.array(data['weight_decays'])
            Z = data['matrix']
            
            if Z.size == 0 or np.isnan(Z).all():
                continue
            

            X, Y = np.meshgrid(weight_decays, dropouts)
            
            
            ax = fig.add_subplot(1, n_models, i, projection='3d')
            axes.append(ax)
            
            
            surf = ax.plot_surface(
                X, Y, Z,
                cmap='viridis_r',
                vmin=z_min, vmax=z_max,
                edgecolor='k',
                linewidth=0.5,
                antialiased=True
            )
            surfaces.append(surf)
            
            ax.set_xlabel("Weight Decay")
            ax.set_ylabel("Dropout")
            ax.set_zlabel("Validation Loss")
            ax.set_title(f"{model}")


            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)


        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(surfaces[0], cax=cbar_ax)
        cbar_ax.tick_params(labelsize=10)

        fig.suptitle(f"Validation Loss Meshgrids - Learning Rate: {lr}", fontsize=14, y=0.98)


        fig.subplots_adjust(left=0.05, right=0.85, top=0.9, bottom=0.1, wspace=0.3)

        plt.show()




def main():
    """Main function to create validation loss meshgrid plots."""
    runs = load_runs("runs.txt")
    run_names = [run['run_id'] for run in runs]
    validation_data = load_validation_data(run_names)
    
    
    valid_runs = [run for run in runs if validation_data[run['run_id']] != float('inf')]
    
    print(f"Found validation data for {len(valid_runs)} runs")
    
    if not valid_runs:
        print("No valid runs found!")
        return
    
    heatmap_data = create_heatmap_data(valid_runs, validation_data)
    plot_meshgrids(heatmap_data)


if __name__ == "__main__":
    main()
