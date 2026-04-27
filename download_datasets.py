"""
Download all benchmark datasets to real_datasets/.
Run locally: python download_datasets.py

Requires: openml, scikit-learn, pandas, numpy
"""
import os
import numpy as np
import pandas as pd
import openml
from sklearn.datasets import load_breast_cancer, load_diabetes

os.makedirs("real_datasets", exist_ok=True)

# --- OpenML datasets (name -> dataset ID) ---
OPENML_IDS = {
    "adult":                  1590,
    "credit-g":               31,
    "california_housing":     43939,
    "heloc":                  45026,
    "heart_disease":          49,
    "australian":             40981,
    "sick":                   38,
    "ilpd":                   1480,
    "blood_transfusion":      1464,
    "chronic_kidney_disease": 43686,
    "gmsc":                   45577,
}

for name, dataset_id in OPENML_IDS.items():
    x_path = f"real_datasets/{name}_X.csv"
    y_path = f"real_datasets/{name}_y.csv"
    if os.path.exists(x_path) and os.path.exists(y_path):
        print(f"[skip]  {name}")
        continue
    print(f"[fetch] {name} (id={dataset_id}) ...", end=" ", flush=True)
    try:
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
        X_df = pd.DataFrame(X, columns=attribute_names)
        pd.DataFrame(X_df).to_csv(x_path, index=False)
        pd.Series(y, name="target").to_csv(y_path, index=False)
        print(f"done  ({len(y)} rows, {X_df.shape[1]} cols)")
    except Exception as e:
        print(f"FAILED: {e}")

# --- sklearn built-ins ---
for name, loader in [("breast_cancer", load_breast_cancer), ("diabetes", load_diabetes)]:
    x_path = f"real_datasets/{name}_X.csv"
    y_path = f"real_datasets/{name}_y.csv"
    if os.path.exists(x_path) and os.path.exists(y_path):
        print(f"[skip]  {name}")
        continue
    print(f"[fetch] {name} (sklearn) ...", end=" ", flush=True)
    bunch = loader(as_frame=True)
    bunch.data.to_csv(x_path, index=False)
    pd.Series(bunch.target.to_numpy(), name="target").to_csv(y_path, index=False)
    print(f"done  ({len(bunch.target)} rows, {bunch.data.shape[1]} cols)")

print("\nDone. Upload the real_datasets/ folder to the server.")
