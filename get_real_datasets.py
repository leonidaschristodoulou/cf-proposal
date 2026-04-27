from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml, fetch_california_housing, load_breast_cancer, load_diabetes

_LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_datasets")


def _load_local(name: str) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """Return (X_df, y_raw) from local CSVs as raw values, or None if not cached."""
    x_path = os.path.join(_LOCAL_DIR, f"{name}_X.csv")
    y_path = os.path.join(_LOCAL_DIR, f"{name}_y.csv")
    if os.path.exists(x_path) and os.path.exists(y_path):
        X_df = pd.read_csv(x_path)
        y_raw = pd.read_csv(y_path)["target"]
        return X_df, y_raw
    return None


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

    # ------------------------------------------------------------------
    if name_key in {"adult", "adult_income"}:
        local = _load_local("adult")
        if local:
            X_df, y_raw = local
        else:
            ds = fetch_openml("adult", version=2, as_frame=as_frame)
            X_df = ds.data
            y_raw = ds.target

        y = (pd.Series(y_raw).astype(str).str.strip() == ">50K").astype(int).to_numpy()

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

    # ------------------------------------------------------------------
    if name_key in {"german_credit", "german", "credit_g", "credit-g", "creditg"}:
        local = _load_local("credit-g")
        if local:
            X_df, y_raw = local
        else:
            ds = fetch_openml("credit-g", version=openml_version, as_frame=as_frame)
            X_df = ds.data
            y_raw = ds.target

        y = (pd.Series(y_raw).astype(str).str.strip() == "good").astype(int).to_numpy()

        num_cols, cat_cols = infer_num_cat_cols(X_df)
        immutables = ImmutableSpec(num={}, cat={"Marital Status"})
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
            meta=meta, immutables=immutables)

    # ------------------------------------------------------------------
    if name_key in {"breast_cancer", "breast-cancer", "cancer"}:
        local = _load_local("breast_cancer")
        if local:
            X_df, y_raw = local
        else:
            bunch = load_breast_cancer(as_frame=True)
            X_df = bunch.data if as_frame else pd.DataFrame(bunch.data, columns=bunch.feature_names)
            y_raw = bunch.target

        y = pd.to_numeric(pd.Series(y_raw), errors="coerce").astype(int).to_numpy()

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

    # ------------------------------------------------------------------
    if name_key in {"diabetes"}:
        local = _load_local("diabetes")
        if local:
            X_df, y_raw = local
            y = pd.to_numeric(pd.Series(y_raw), errors="coerce").astype(int).to_numpy()
            thr = None
        else:
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
            "threshold": thr if not local else None,
        }
        return LoadedDataset(
            name="diabetes_binarized",
            X_df=X_df, y=y,
            num_cols=num_cols, cat_cols=cat_cols,
            meta=meta, immutables=ImmutableSpec.empty())

    # ------------------------------------------------------------------
    if name_key in {"california_housing", "california", "ca_housing", "housing_ca"}:
        local = _load_local("california_housing")
        if local:
            X_df, y_raw = local
            y_float = pd.to_numeric(pd.Series(y_raw), errors="coerce").to_numpy(dtype=float)
            thr = float(np.median(y_float))
            y = (y_float >= thr).astype(int)
        else:
            ds = fetch_openml(name="california_housing", version=1, as_frame=as_frame)
            X_df = ds.data
            y_float = ds.target.to_numpy(dtype=float)
            thr = float(np.median(y_float))
            y = (y_float >= thr).astype(int)

        num_cols, cat_cols = infer_num_cat_cols(X_df)
        meta = {
            "source": "openml",
            "sklearn_name": "fetch_california_housing",
            "task": "binary_classification_via_binarization",
            "binarize_rule": "median",
            "threshold": thr,
            "original_target": "MedHouseVal",
        }
        immutables = ImmutableSpec(num={"Latitude", "Longitude", "HouseAge"}, cat=set())
        return LoadedDataset(
            name="california_housing_binarized",
            X_df=X_df, y=y,
            num_cols=num_cols, cat_cols=cat_cols,
            meta=meta, immutables=immutables,
        )

    # ------------------------------------------------------------------
    if name_key in {"heloc", "fico_heloc", "fico-heloc"}:
        local = _load_local("heloc")
        if local:
            X_df, y_raw = local
        else:
            ds = fetch_openml(name="HELOC", version=1, as_frame=as_frame)
            X_df = ds.data
            y_raw = ds.target

        target = pd.Series(y_raw).astype(str).str.strip().str.lower()
        uniques = set(target.unique())
        if uniques <= {"0", "1"}:
            y = target.astype(int).to_numpy()
        elif {"good", "bad"}.issubset(uniques):
            y = (target == "good").astype(int).to_numpy()
        else:
            numeric = pd.to_numeric(target, errors="coerce")
            if numeric.notna().all():
                y = numeric.astype(int).to_numpy()
            else:
                codes, _ = pd.factorize(target, sort=True)
                y = codes.astype(int)

        num_cols, cat_cols = infer_num_cat_cols(X_df)
        meta = {
            "source": "openml",
            "openml_name": "HELOC",
            "openml_version": 1,
            "task": "binary_classification",
            "positive_label": "Good",
        }
        immutables = ImmutableSpec(num={"ExternalRiskEstimate"}, cat=set())
        return LoadedDataset(
            name="heloc",
            X_df=X_df, y=y,
            num_cols=num_cols, cat_cols=cat_cols,
            meta=meta, immutables=immutables,
        )

    # ------------------------------------------------------------------
    if name_key in {"heart_disease", "heart-disease", "heart", "heart_cleveland", "cleveland"}:
        local = _load_local("heart_disease")
        if local:
            X_df, y_raw = local
        else:
            ds = fetch_openml("heart-c", version=1, as_frame=as_frame)
            X_df = ds.data
            y_raw = ds.target

        raw = pd.Series(y_raw).astype(str).str.strip()
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.notna().all():
            y_all = (numeric.to_numpy() > 0).astype(int)
        else:
            y_all = raw.str.startswith(">").to_numpy().astype(int)

        mask = X_df.notna().all(axis=1).to_numpy()
        X_df = X_df.iloc[mask].reset_index(drop=True)
        y = y_all[mask]

        num_cols, cat_cols = infer_num_cat_cols(X_df)
        meta = {
            "source": "openml",
            "openml_name": "heart-c",
            "openml_version": 1,
            "task": "binary_classification_via_binarization",
            "binarize_rule": "target > 0",
            "positive_label": "disease (1-4)",
        }
        immutables = ImmutableSpec(num={"age"}, cat={"sex"})
        return LoadedDataset(
            name="heart_disease",
            X_df=X_df, y=y,
            num_cols=num_cols, cat_cols=cat_cols,
            meta=meta, immutables=immutables,
        )

    # ------------------------------------------------------------------
    if name_key in {"australian", "aus", "australian_credit"}:
        local = _load_local("australian")
        if local:
            X_df, y_raw = local
        else:
            ds = fetch_openml("Australian", version=2, as_frame="auto")
            X_df = pd.DataFrame(ds.data.toarray() if hasattr(ds.data, "toarray") else ds.data,
                                columns=ds.feature_names)
            y_raw = ds.target

        raw = pd.Series(y_raw).astype(str).str.strip()
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.notna().all():
            y = (numeric.to_numpy() > 0).astype(int)
        else:
            codes, _ = pd.factorize(raw, sort=True)
            y = codes.astype(int)

        num_cols, cat_cols = infer_num_cat_cols(X_df)
        meta = {
            "source": "openml",
            "openml_name": "Australian",
            "openml_version": 1,
            "task": "binary_classification",
        }
        immutables = ImmutableSpec(num=set(), cat=set())
        return LoadedDataset(
            name="australian",
            X_df=X_df, y=y,
            num_cols=num_cols, cat_cols=cat_cols,
            meta=meta, immutables=immutables,
        )

    # ------------------------------------------------------------------
    if name_key in {"sick", "sick_thyroid", "thyroid_sick"}:
        local = _load_local("sick")
        if local:
            X_df, y_raw = local
        else:
            ds = fetch_openml("sick", version=1, as_frame="auto")
            X_df = pd.DataFrame(ds.data.toarray() if hasattr(ds.data, "toarray") else ds.data,
                                columns=ds.feature_names)
            y_raw = ds.target

        y_all = (pd.Series(y_raw).astype(str).str.strip() == "sick").astype(int).to_numpy()

        mask = X_df.notna().mean(axis=1).to_numpy() >= 0.5
        X_df = X_df.iloc[mask].reset_index(drop=True)
        y = y_all[mask]
        X_df = X_df.dropna(axis=1, how="all")

        num_cols, cat_cols = infer_num_cat_cols(X_df)
        meta = {
            "source": "openml",
            "openml_name": "sick",
            "openml_version": 1,
            "task": "binary_classification",
            "positive_label": "sick",
        }
        immutables = ImmutableSpec(num={"age"}, cat={"sex"})
        return LoadedDataset(
            name="sick",
            X_df=X_df, y=y,
            num_cols=num_cols, cat_cols=cat_cols,
            meta=meta, immutables=immutables,
        )

    # ------------------------------------------------------------------
    if name_key in {"ilpd", "indian_liver", "indian_liver_patient"}:
        local = _load_local("ilpd")
        if local:
            X_df, y_raw = local
        else:
            ds = fetch_openml("ilpd", version=1, as_frame="auto")
            X_df = pd.DataFrame(ds.data.toarray() if hasattr(ds.data, "toarray") else ds.data,
                                columns=ds.feature_names)
            y_raw = ds.target

        raw = pd.Series(y_raw).astype(str).str.strip()
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.notna().all():
            y_all = (numeric.to_numpy() == 1).astype(int)
        else:
            y_all = (raw == "1").astype(int).to_numpy()

        mask = X_df.notna().mean(axis=1).to_numpy() >= 0.5
        X_df = X_df.iloc[mask].reset_index(drop=True)
        y = y_all[mask]

        num_cols, cat_cols = infer_num_cat_cols(X_df)
        meta = {
            "source": "openml",
            "openml_name": "ilpd",
            "openml_version": 1,
            "task": "binary_classification",
            "positive_label": "1 (liver patient)",
        }
        immutables = ImmutableSpec(num={"Age"}, cat={"Gender"})
        return LoadedDataset(
            name="ilpd",
            X_df=X_df, y=y,
            num_cols=num_cols, cat_cols=cat_cols,
            meta=meta, immutables=immutables,
        )

    # ------------------------------------------------------------------
    if name_key in {"blood_transfusion", "blood-transfusion", "blood", "transfusion"}:
        local = _load_local("blood_transfusion")
        if local:
            X_df, y_raw = local
        else:
            ds = fetch_openml(data_id=1464, as_frame="auto")
            X_df = pd.DataFrame(ds.data.toarray() if hasattr(ds.data, "toarray") else ds.data,
                                columns=ds.feature_names)
            y_raw = ds.target

        raw = pd.Series(y_raw).astype(str).str.strip()
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.notna().all():
            y = (numeric.to_numpy() == 1).astype(int)
        else:
            y = (raw == "1").astype(int).to_numpy()

        num_cols, cat_cols = infer_num_cat_cols(X_df)
        meta = {
            "source": "openml",
            "openml_data_id": 1464,
            "openml_name": "blood-transfusion-service-center",
            "task": "binary_classification",
            "positive_label": "1 (donated)",
        }
        immutables = ImmutableSpec(num=set(), cat=set())
        return LoadedDataset(
            name="blood_transfusion",
            X_df=X_df, y=y,
            num_cols=num_cols, cat_cols=cat_cols,
            meta=meta, immutables=immutables,
        )

    # ------------------------------------------------------------------
    if name_key in {"chronic_kidney", "chronic_kidney_disease", "ckd", "kidney"}:
        local = _load_local("chronic_kidney_disease")
        if local:
            X_df, y_raw = local
        else:
            ds = fetch_openml(data_id=43686, as_frame="auto")
            X_df = pd.DataFrame(ds.data.toarray() if hasattr(ds.data, "toarray") else ds.data,
                                columns=ds.feature_names)
            y_raw = ds.target

        raw = pd.Series(y_raw)
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.notna().all():
            y_all = numeric.astype(int).to_numpy()
        else:
            y_all = (raw.astype(str).str.strip().str.lower() == "ckd").astype(int).to_numpy()

        mask = X_df.notna().mean(axis=1).to_numpy() >= 0.5
        X_df = X_df.iloc[mask].reset_index(drop=True)
        y = y_all[mask]

        num_cols, cat_cols = infer_num_cat_cols(X_df)
        meta = {
            "source": "openml",
            "openml_name": "chronic-kidney-disease",
            "openml_version": 2,
            "openml_data_id": 43686,
            "task": "binary_classification",
            "positive_label": "ckd",
        }
        immutables = ImmutableSpec(num={"age"}, cat={"htn", "dm", "cad"})
        return LoadedDataset(
            name="chronic_kidney_disease",
            X_df=X_df, y=y,
            num_cols=num_cols, cat_cols=cat_cols,
            meta=meta, immutables=immutables,
        )

    # ------------------------------------------------------------------
    if name_key in {"gmsc", "give_me_some_credit", "give-me-some-credit"}:
        local = _load_local("gmsc")
        if local:
            X_df, y_raw = local
        else:
            ds = fetch_openml(data_id=45577, as_frame=as_frame)
            X_df = ds.data
            y_raw = ds.target

        y_s = pd.to_numeric(pd.Series(y_raw), errors="ignore")
        if not np.issubdtype(y_s.dtype, np.number):
            y_s = pd.Series(pd.factorize(y_s)[0])
        y = y_s.to_numpy().astype(int)

        num_cols, cat_cols = infer_num_cat_cols(X_df)
        meta = {
            "source": "openml",
            "openml_data_id": 45577,
            "openml_name": "Give-Me-Some-Credit",
            "task": "binary_classification",
        }
        imm_num = {"age"}
        extra_hist = {
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
        }
        imm_num |= {c for c in extra_hist if c in X_df.columns}
        immutables = ImmutableSpec(num=imm_num, cat=set())
        return LoadedDataset(
            name="gmsc",
            X_df=X_df, y=y,
            num_cols=num_cols, cat_cols=cat_cols,
            meta=meta, immutables=immutables,
        )

    raise ValueError(
        f"Unknown dataset name: {name}. Supported: adult, credit-g, breast_cancer, diabetes, "
        "california_housing, chronic_kidney_disease, gmsc, heloc, heart_disease, australian, sick, ilpd, blood_transfusion."
    )


def load_many(names: List[str], **kwargs) -> List[LoadedDataset]:
    """Convenience to load a list of datasets with shared kwargs."""
    return [load_classification_dataset(n, **kwargs) for n in names]
