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
    
    if name_key in {"heloc", "fico_heloc", "fico-heloc"}:
        # FICO HELOC dataset — all numeric, binary classification
        ds = fetch_openml(name="HELOC", version=1, as_frame=as_frame)
        X_df = ds.data

        # Target: "Good" = 1 (repaid), "Bad" = 0 (default).
        # The OpenML encoding varies: may be strings ("Good"/"Bad"), lower-case,
        # or already 1/0 integers — handle all cases.
        raw = ds.target
        target = raw.astype(str).str.strip().str.lower()
        uniques = set(target.unique())
        if uniques <= {"0", "1"}:
            # Already binary-encoded as strings "0"/"1"
            y = target.astype(int).to_numpy()
        elif {"good", "bad"}.issubset(uniques):
            y = (target == "good").astype(int).to_numpy()
        elif np.issubdtype(raw.dtype, np.number):
            y = raw.astype(int).to_numpy()
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
        # ExternalRiskEstimate is an external credit score — not actionable by the applicant
        immutables = ImmutableSpec(num={"ExternalRiskEstimate"}, cat=set())
        return LoadedDataset(
            name="heloc",
            X_df=X_df,
            y=y,
            num_cols=num_cols,
            cat_cols=cat_cols,
            meta=meta,
            immutables=immutables,
        )

    if name_key in {"heart_disease", "heart-disease", "heart", "heart_cleveland", "cleveland"}:
        # UCI Cleveland Heart Disease via OpenML ("heart-c")
        ds = fetch_openml("heart-c", version=1, as_frame=as_frame)
        X_df = ds.data

        # Target: '<50' = no disease (0), '>50_1' = disease (1)
        # Use numpy-level ops to avoid any pandas index alignment surprises.
        raw = ds.target.astype(str).str.strip()
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.notna().all():
            y_all = (numeric.to_numpy() > 0).astype(int)
        else:
            # String labels: positive class starts with '>' (i.e. '>50_1')
            y_all = raw.str.startswith(">").to_numpy().astype(int)

        # Drop the ~7 rows with missing values in X — use numpy mask to
        # avoid any index alignment issues between X_df and y_all.
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
        # Age and sex are immutable demographic attributes
        immutables = ImmutableSpec(num={"age"}, cat={"sex"})
        return LoadedDataset(
            name="heart_disease",
            X_df=X_df,
            y=y,
            num_cols=num_cols,
            cat_cols=cat_cols,
            meta=meta,
            immutables=immutables,
        )

    if name_key in {"australian", "aus", "australian_credit"}:
        # AutoML benchmark: Statlog (Australian Credit Approval) via OpenML
        # Use 'auto' because this dataset is stored as sparse ARFF on OpenML
        ds = fetch_openml("Australian", version=1, as_frame="auto")
        X_df = pd.DataFrame(ds.data.toarray() if hasattr(ds.data, "toarray") else ds.data,
                            columns=ds.feature_names)

        raw = pd.Series(ds.target).astype(str).str.strip()
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
        # A6 (employment status) and A8 (years at job) are demographic/historical
        immutables = ImmutableSpec(num=set(), cat=set())
        return LoadedDataset(
            name="australian",
            X_df=X_df,
            y=y,
            num_cols=num_cols,
            cat_cols=cat_cols,
            meta=meta,
            immutables=immutables,
        )

    if name_key in {"sick", "sick_thyroid", "thyroid_sick"}:
        # AutoML benchmark: Thyroid sick dataset via OpenML
        ds = fetch_openml("sick", version=1, as_frame="auto")
        X_df = pd.DataFrame(ds.data.toarray() if hasattr(ds.data, "toarray") else ds.data,
                            columns=ds.feature_names)

        raw = pd.Series(ds.target).astype(str).str.strip()
        y = (raw == "sick").astype(int).to_numpy()

        # Drop rows with all-NaN or majority-NaN (sick has some missing values)
        mask = X_df.notna().mean(axis=1).to_numpy() >= 0.5
        X_df = X_df.iloc[mask].reset_index(drop=True)
        y = y[mask]

        num_cols, cat_cols = infer_num_cat_cols(X_df)
        meta = {
            "source": "openml",
            "openml_name": "sick",
            "openml_version": 1,
            "task": "binary_classification",
            "positive_label": "sick",
        }
        # Age and sex are demographic immutables
        immutables = ImmutableSpec(num={"age"}, cat={"sex"})
        return LoadedDataset(
            name="sick",
            X_df=X_df,
            y=y,
            num_cols=num_cols,
            cat_cols=cat_cols,
            meta=meta,
            immutables=immutables,
        )

    if name_key in {"ilpd", "indian_liver", "indian_liver_patient"}:
        # AutoML benchmark: Indian Liver Patient Dataset via OpenML
        ds = fetch_openml("ilpd", version=1, as_frame="auto")
        X_df = pd.DataFrame(ds.data.toarray() if hasattr(ds.data, "toarray") else ds.data,
                            columns=ds.feature_names)

        raw = pd.Series(ds.target).astype(str).str.strip()
        # Label "1" = liver patient (positive), "2" = no disease
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.notna().all():
            y = (numeric.to_numpy() == 1).astype(int)
        else:
            y = (raw == "1").astype(int).to_numpy()

        # Drop rows missing target or with >50% missing features
        mask = X_df.notna().mean(axis=1).to_numpy() >= 0.5
        X_df = X_df.iloc[mask].reset_index(drop=True)
        y = y[mask]

        num_cols, cat_cols = infer_num_cat_cols(X_df)
        meta = {
            "source": "openml",
            "openml_name": "ilpd",
            "openml_version": 1,
            "task": "binary_classification",
            "positive_label": "1 (liver patient)",
        }
        # Age and gender are immutable demographics
        immutables = ImmutableSpec(num={"Age"}, cat={"Gender"})
        return LoadedDataset(
            name="ilpd",
            X_df=X_df,
            y=y,
            num_cols=num_cols,
            cat_cols=cat_cols,
            meta=meta,
            immutables=immutables,
        )

    if name_key in {"blood_transfusion", "blood-transfusion", "blood", "transfusion"}:
        # AutoML benchmark: Blood Transfusion Service Center (OpenML id 1464)
        ds = fetch_openml(data_id=1464, as_frame="auto")
        X_df = pd.DataFrame(ds.data.toarray() if hasattr(ds.data, "toarray") else ds.data,
                            columns=ds.feature_names)

        raw = pd.Series(ds.target).astype(str).str.strip()
        # Label "1" = donated blood, "2" = did not donate
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
        # All features are behavioural (recency, frequency, monetary, time) — none immutable
        immutables = ImmutableSpec(num=set(), cat=set())
        return LoadedDataset(
            name="blood_transfusion",
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
        f"Unknown dataset name: {name}. Supported: adult, credit-g, breast_cancer, diabetes, "
        "california_housing, gmsc, heloc, heart_disease, australian, sick, ilpd, blood_transfusion."
    )


def load_many(names: List[str], **kwargs) -> List[LoadedDataset]:
    """Convenience to load a list of datasets with shared kwargs."""
    return [load_classification_dataset(n, **kwargs) for n in names]