from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml, fetch_california_housing, load_breast_cancer, load_diabetes

@dataclass(frozen=True)
class ImmutableSpec:
    num: Set[str]
    cat: Set[str]

    @staticmethod
    def empty() -> "ImmutableSpec":
        return ImmutableSpec(num=set(), cat=set())

@dataclass(frozen=True)
class LoadedDataset:
    name: str
    X_df: pd.DataFrame
    y: np.ndarray
    num_cols: List[str]
    cat_cols: List[str]
    meta: Dict[str, Any]
    immutables: ImmutableSpec = ImmutableSpec.empty()


def infer_num_cat_cols(X_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Basic, reproducible split: numbers vs everything else."""
    num_cols = X_df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]
    return num_cols, cat_cols


def load_classification_dataset(
    name: str,
    *,
    as_frame: bool = True,
    openml_version: str = 'active',
    diabetes_binarize: str = "median",  # "median" or "threshold"
    diabetes_threshold: float = 140.0,  # ignored unless diabetes_binarize="threshold"
) -> LoadedDataset:
    """
    Load common benchmark datasets in a consistent format.

    Supported:
      - "adult"         (OpenML, mixed types, binary)
      - "credit-g"      (OpenML German Credit, mixed types, binary)
      - "breast_cancer" (sklearn, numeric, binary)
      - "diabetes"      (sklearn regression -> binarized for classification)

    Returns:
      LoadedDataset with X_df, y, num_cols, cat_cols, meta.
    """
    name_key = name.strip().lower().replace(" ", "_")

    if name_key in {"adult", "adult_income"}:
        ds = fetch_openml("adult", version=2, as_frame=as_frame)
        X_df = ds.data
        # Adult target is string labels like ">50K" / "<=50K"
        y = (ds.target == ">50K").astype(int).to_numpy()

        num_cols, cat_cols = infer_num_cat_cols(X_df)
        immutables = ImmutableSpec(num={"age", "fnlwgt"}, cat={"sex", "race"})
        meta = {
            "source": "openml",
            "openml_name": "adult",
            "openml_version": openml_version or 2,
            "task": "binary_classification",
            "positive_label": ">50K",
        }
        return LoadedDataset(
            name="adult", X_df=X_df, y=y, 
            num_cols=num_cols, cat_cols=cat_cols, 
            meta=meta, immutables=immutables)

    if name_key in {"german_credit", "german", "credit_g", "credit-g", "creditg"}:
        # OpenML canonical name is "credit-g"
        ds = fetch_openml("credit-g", version=openml_version, as_frame=as_frame)
        X_df = ds.data

        # target is typically "good"/"bad" (strings) -> map to 1/0
        # We'll treat "good" as positive by default.
        target = ds.target.astype(str)
        y = (target == "good").astype(int).to_numpy()

        num_cols, cat_cols = infer_num_cat_cols(X_df)
        immutables = ImmutableSpec(num={}, cat={"Marital Status"}) #ImmutableSpec.empty()
        meta = {
            "source": "openml",
            "openml_name": "credit-g",
            "openml_version": openml_version,
            "task": "binary_classification",
            "positive_label": "good",
        }
        return LoadedDataset(
            name="credit-g", X_df=X_df, y=y, 
            num_cols=num_cols, cat_cols=cat_cols, 
            meta=meta,immutables=immutables)

    if name_key in {"breast_cancer", "breast-cancer", "cancer"}:
        bunch = load_breast_cancer(as_frame=True)
        X_df = bunch.data if as_frame else pd.DataFrame(bunch.data, columns=bunch.feature_names)
        y = bunch.target.astype(int)

        num_cols, cat_cols = infer_num_cat_cols(X_df)
        meta = {
            "source": "sklearn",
            "sklearn_name": "load_breast_cancer",
            "task": "binary_classification",
        }
        return LoadedDataset(
            name="breast_cancer", X_df=X_df, y=y, 
            num_cols=num_cols, cat_cols=cat_cols, 
            meta=meta, immutables=ImmutableSpec.empty())

    if name_key in {"diabetes"}:
        # NOTE: sklearn diabetes is regression
        bunch = load_diabetes(as_frame=True)
        X_df = bunch.data if as_frame else pd.DataFrame(bunch.data, columns=bunch.feature_names)
        y_reg = bunch.target.to_numpy()

        if diabetes_binarize == "median":
            thr = float(np.median(y_reg))
        elif diabetes_binarize == "threshold":
            thr = float(diabetes_threshold)
        else:
            raise ValueError("diabetes_binarize must be 'median' or 'threshold'.")

        y = (y_reg >= thr).astype(int)

        num_cols, cat_cols = infer_num_cat_cols(X_df)
        meta = {
            "source": "sklearn",
            "sklearn_name": "load_diabetes",
            "task": "binary_classification_via_binarization",
            "binarize_rule": diabetes_binarize,
            "threshold": thr,
        }
        return LoadedDataset(
            name="diabetes_binarized", 
            X_df=X_df, y=y, 
            num_cols=num_cols, cat_cols=cat_cols, 
            meta=meta, immutables=ImmutableSpec.empty())
    
    if name_key in {"california_housing", "california", "ca_housing", "housing_ca"}:
        ds = fetch_openml(
            name="california_housing",
            version=1,
            as_frame=as_frame,
        )
        X_df = ds.data
        y_reg = ds.target.to_numpy(dtype=float)

        # Binarize: high value vs low value (median split)
        thr = float(np.median(y_reg))
        y = (y_reg >= thr).astype(int)

        num_cols, cat_cols = infer_num_cat_cols(X_df)
        meta = {
            "source": "openml",
            "sklearn_name": "fetch_california_housing",
            "task": "binary_classification_via_binarization",
            "binarize_rule": "median",
            "threshold": thr,
            "original_target": "MedHouseVal",
        }

        # Strong, defensible immutables: location + age (cannot reduce age)
        immutables = ImmutableSpec(num={"Latitude", "Longitude", "HouseAge"}, cat=set())

        return LoadedDataset(
            name="california_housing_binarized",
            X_df=X_df,
            y=y,
            num_cols=num_cols,
            cat_cols=cat_cols,
            meta=meta,
            immutables=immutables,
        )
    
    if name_key in {"gmsc", "give_me_some_credit", "give-me-some-credit"}:
        # OpenML dataset "Give-Me-Some-Credit" (id=45577)
        ds = fetch_openml(data_id=45577, as_frame=as_frame)  # :contentReference[oaicite:2]{index=2}
        X_df = ds.data
        y = ds.target

        # Make target numeric {0,1}
        if isinstance(y, pd.Series):
            # If already numeric, this is a no-op; if strings, factorize consistently.
            y = pd.to_numeric(y, errors="ignore")
            if not np.issubdtype(y.dtype, np.number):
                y = pd.Series(pd.factorize(y)[0], index=y.index)
            y = y.to_numpy()
        y = y.astype(int)

        num_cols, cat_cols = infer_num_cat_cols(X_df)
        meta = {
            "source": "openml",
            "openml_data_id": 45577,
            "openml_name": "Give-Me-Some-Credit",
            "task": "binary_classification",
        }

        # Typical immutables (adjust to your philosophy):
        # - Age is immutable
        # - Past-due counters are *historical* (not realistically changeable)
        #   but you might prefer "immutable" or "only allowed to decrease" (bounded/monotone).
        imm_num = {"age"}  # keep minimal to avoid column-name surprises
        # If these columns exist (names can vary by version), you can include them:
        extra_hist = {
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
        }
        imm_num |= {c for c in extra_hist if c in X_df.columns}

        immutables = ImmutableSpec(num=imm_num, cat=set())

        return LoadedDataset(
            name="gmsc",
            X_df=X_df,
            y=y,
            num_cols=num_cols,
            cat_cols=cat_cols,
            meta=meta,
            immutables=immutables,
        )

    raise ValueError(
        f"Unknown dataset name: {name}. Supported: adult, credit-g (german credit), breast_cancer, diabetes."
    )


def load_many(names: List[str], **kwargs) -> List[LoadedDataset]:
    """Convenience to load a list of datasets with shared kwargs."""
    return [load_classification_dataset(n, **kwargs) for n in names]