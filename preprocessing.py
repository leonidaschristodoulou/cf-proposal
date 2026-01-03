import numpy as np

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def make_preprocessor(num_cols, cat_cols, *, scale_num=True, ohe_sparse=False):
    transformers = []

    if num_cols:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_num:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(num_steps), num_cols))

    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=ohe_sparse)),
            ]),
            cat_cols
        ))

    if not transformers:
        raise ValueError("No columns to transform (num_cols and cat_cols are both empty).")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_feature_info_from_pre(pre, num_cols, cat_cols):
    """
    Returns:
      num_idx: indices for numeric features in transformed space
      cat_groups: list of arrays, one per categorical feature, containing the one-hot indices
    """
    # numeric block length: for this pipeline, it equals len(num_cols) if num exists
    num_dim = len(num_cols) if "num" in pre.named_transformers_ else 0
    num_idx = np.arange(num_dim, dtype=int)

    cat_groups = []
    if "cat" in pre.named_transformers_:
        ohe = pre.named_transformers_["cat"].named_steps["ohe"]
        cats = ohe.categories_  # list aligned with cat_cols

        offset = num_dim
        for levels in cats:
            g = np.arange(offset, offset + len(levels), dtype=int)
            cat_groups.append(g)
            offset += len(levels)

    return num_idx, cat_groups


def safe_immutable_num(num_cols, num_idx, immutable_num_names):
    """Map immutable numeric feature names to transformed indices; ignore missing."""
    out = set()
    for name in immutable_num_names:
        if name in num_cols:
            out.add(int(num_idx[num_cols.index(name)]))
    return frozenset(out)


def safe_immutable_cat(cat_cols, immutable_cat_names):
    """Immutable categorical features are represented as group indices (0..len(cat_cols)-1); ignore missing."""
    return frozenset(i for i, c in enumerate(cat_cols) if c in immutable_cat_names)
    

def inverse_preprocess(x, pre, num_cols=None, cat_cols=None):
    """
    Decode a model-space vector back to human-readable features.

    - Numeric: invert scaling ONLY (imputation is ignored)
    - Categorical: invert one-hot encoding ONLY

    Notes:
    - Uses the FITTED ColumnTransformer (`pre.transformers_`) as the source of truth
      for column order and block sizes. The passed num_cols/cat_cols are treated
      as optional filters (useful if you want to restrict the output).
    - Works for dense or scipy sparse inputs.
    """
    # ---- ensure 2D without breaking sparse ----
    if hasattr(x, "shape"):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        # if already (1, d) keep it
    else:
        x = np.asarray(x).reshape(1, -1)

    # ---- get actual fitted column lists and order ----
    fitted_cols = {}
    for name, _, cols in pre.transformers_:
        if name == "remainder":
            continue
        # cols can be array-like, slice, boolean mask; assume list/Index for your case
        fitted_cols[name] = list(cols)

    out = {}
    offset = 0

    # ---- numeric block ----
    if "num" in pre.named_transformers_:
        num_cols_used = fitted_cols.get("num", [])
        num_dim = len(num_cols_used)

        X_num = x[:, offset:offset + num_dim]
        # scaler expects dense
        if hasattr(X_num, "toarray"):
            X_num_dense = X_num.toarray()
        else:
            X_num_dense = np.asarray(X_num)

        num_pipe = pre.named_transformers_["num"]
        scaler = num_pipe.named_steps.get("scaler", None)

        if scaler is not None:
            num_orig = scaler.inverse_transform(X_num_dense)[0]
        else:
            num_orig = X_num_dense[0]

        # optional filter: only return requested num_cols
        if num_cols is not None:
            keep = [c for c in num_cols_used if c in set(num_cols)]
        else:
            keep = num_cols_used

        if keep:
            idx_map = {c: i for i, c in enumerate(num_cols_used)}
            out.update({c: float(num_orig[idx_map[c]]) for c in keep})

        offset += num_dim

    # ---- categorical block ----
    if "cat" in pre.named_transformers_:
        cat_cols_used = fitted_cols.get("cat", [])
        cat_pipe = pre.named_transformers_["cat"]
        ohe = cat_pipe.named_steps["ohe"]

        X_cat = x[:, offset:]
        cat_orig = ohe.inverse_transform(X_cat)[0]  # shape (n_cat,)

        # optional filter: only return requested cat_cols
        if cat_cols is not None:
            keep = [c for c in cat_cols_used if c in set(cat_cols)]
        else:
            keep = cat_cols_used

        if keep:
            idx_map = {c: i for i, c in enumerate(cat_cols_used)}
            out.update({c: cat_orig[idx_map[c]] for c in keep})

    return out