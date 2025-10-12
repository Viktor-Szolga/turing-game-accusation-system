import math
from typing import Optional
import os

def load_runs(filepath: str):
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


def query_runs(
    runs,
    model: Optional[str] = None,
    lr: Optional[float] = None,
    dropout: Optional[float] = None,
    weight_decay: Optional[float] = None,
    lr_min: Optional[float] = None,
    lr_max: Optional[float] = None,
    dropout_min: Optional[float] = None,
    dropout_max: Optional[float] = None,
    weight_decay_min: Optional[float] = None,
    weight_decay_max: Optional[float] = None,
    allow_linear: Optional[bool] = False
):
    results = []
    for run in runs:
        if allow_linear is False:
            if run["model"].lower() == "model l":
                continue
        # model match
        if model is not None and run["model"].lower() != model.lower():
            continue

        # exact matches
        if lr is not None and not math.isclose(run["lr"], lr, rel_tol=1e-9, abs_tol=1e-12):
            continue
        if dropout is not None and not math.isclose(run["dropout"], dropout, rel_tol=1e-9, abs_tol=1e-12):
            continue
        if weight_decay is not None and not math.isclose(run["weight_decay"], weight_decay, rel_tol=1e-9, abs_tol=1e-12):
            continue

        # ranges
        if lr_min is not None and run["lr"] < lr_min:
            continue
        if lr_max is not None and run["lr"] > lr_max:
            continue
        if dropout_min is not None and run["dropout"] < dropout_min:
            continue
        if dropout_max is not None and run["dropout"] > dropout_max:
            continue
        if weight_decay_min is not None and run["weight_decay"] < weight_decay_min:
            continue
        if weight_decay_max is not None and run["weight_decay"] > weight_decay_max:
            continue

        results.append(run)

    return results


def format_results(runs):
    run_names = [r["run_id"] for r in runs]
    legends = [f'{r["model"].split()[-1]} | lr={r["lr"]} | Dropout={r["dropout"]} | Weight decay={r["weight_decay"]}' for r in runs]

    print("```python")
    print("run_names = [")
    for r in run_names:
        print(f'    "{r}",')
    print(f'    "run138",')
    print("]\n")
    print("legends = [")
    for l in legends:
        print(f'    "{l}",')
    print(f'    "best",')
    print("]")
    print("```")


if __name__ == "__main__":
    root_dir = "gridsearch"
    runs = load_runs(os.path.join(root_dir, "runs.txt"))

    # Example: Model C with dropout between 0.5 and 0.75, any lr, weight_decay fixed
    res = query_runs(
        runs,
        model="Model B",
        #dropout_min=0.65,
        #dropout_max=0.85,
        dropout=0.85,
        weight_decay=0
        #weight_decay=0
    )

    format_results(res)
