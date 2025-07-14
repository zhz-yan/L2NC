import os
import json
import pandas as pd
import numpy as np


DATASET = "plaid"
root_dir = f"{DATASET}/"
n_classes = 11  # classes 0 to 11

all_results = {
    "F1_unknown": {i: [] for i in range(n_classes)},
    "F_macro": {i: [] for i in range(n_classes)},
    "AUROC": {i: [] for i in range(n_classes)},
}

for unknown_class in range(n_classes):
    folder_path = os.path.join(root_dir, str(unknown_class))
    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            json_path = os.path.join(folder_path, file)
            with open(json_path, "r") as f:
                content = json.load(f)

                best_f1 = -1
                best_metrics = None
                for tau, metrics in content.items():
                    if metrics["F1_unknown"] > best_f1:
                        best_f1 = metrics["F1_unknown"]
                        best_metrics = metrics

                if best_metrics:
                    all_results["F1_unknown"][unknown_class].append(best_metrics["F1_unknown"] * 100)
                    all_results["F_macro"][unknown_class].append(best_metrics["F_macro"] * 100)
                    all_results["AUROC"][unknown_class].append(best_metrics["AUROC"] * 100)

df_rows = {}

for metric in ["F1_unknown", "F_macro", "AUROC"]:
    row = {}
    for cls in range(n_classes):
        values = all_results[metric][cls]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            row[cls] = f"{mean:.1f} ± {std:.1f}"
        else:
            row[cls] = "-"
    df_rows[metric] = row

df = pd.DataFrame.from_dict(df_rows, orient="index")
df.index.name = "Metric"

avg_col = {}
for metric in ["F1_unknown", "F_macro", "AUROC"]:
    values = [
        float(cell.split("±")[0]) for cell in df.loc[metric]
        if isinstance(cell, str) and "±" in cell # mean  ± std
    ]
    avg_col[metric] = f"{np.mean(values):.1f}"

df["Avg."] = pd.Series(avg_col)

df.to_excel(f"case_1_{DATASET}.xlsx")
