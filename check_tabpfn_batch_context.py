# ==========================================================================
# Live check: does TabPFN's prediction for a NICE-selected counterfactual
# candidate change depending on whether it's scored as part of a full-batch
# call (matching NICE's internal justified_cf check, done once over the
# whole training set at build time) vs. scored alone (matching this
# benchmark's own re-verification in nice_cf_one)?
#
# Reproduces one specific case: australian, TabPFN, seed=1, idx=2 -- one of
# the 36 instances where nice_base returned status="not_flipped" on exactly
# the set PACE's relaxation recovers.
# ==========================================================================

import sys
import numpy as np

sys.path.insert(0, "/nvme/h/lchristodoulou/pace/cf-proposal/")
from get_real_datasets import load_many, ImmutableSpec
from preprocessing import make_preprocessor, build_feature_info_from_pre, safe_immutable_num, safe_immutable_cat
from pfn_cf_guided_onehot import FeatureInfo
from sklearn.model_selection import train_test_split
from nice import NICE

SEED = 1
TE_IDX = 2
DATASET = "australian"

print(f"===== {DATASET} / TabPFN / seed={SEED} / idx={TE_IDX} (nice_base, status was not_flipped) =====", flush=True)

ds = load_many([DATASET], openml_version=2)[0]
X_df = ds.X_df
y = np.asarray(ds.y).astype(int)
num_cols, cat_cols = list(ds.num_cols), list(ds.cat_cols)
print("num_cols:", len(num_cols), "cat_cols:", len(cat_cols), flush=True)

idx = np.arange(len(X_df))
idx_tr, idx_te = train_test_split(idx, test_size=0.25, stratify=y, random_state=SEED)
X_tr_raw = X_df.iloc[idx_tr].copy()
X_te_raw = X_df.iloc[idx_te].copy()
y_tr, y_te = y[idx_tr], y[idx_te]

pre = make_preprocessor(num_cols, cat_cols, ohe_sparse=False)
pre.fit(X_tr_raw)
X_tr_sc = np.asarray(pre.transform(X_tr_raw), dtype=np.float64)
X_te_sc = np.asarray(pre.transform(X_te_raw), dtype=np.float64)

from tabpfn import TabPFNClassifier
model = TabPFNClassifier(device="auto").fit(X_tr_sc, y_tr)


def predict_fn(X):
    return model.predict_proba(np.asarray(X, dtype=np.float64))


nice_obj = NICE(
    X_train=X_tr_sc.copy(), y_train=y_tr, predict_fn=predict_fn,
    cat_feat=[], num_feat="auto", distance_metric="HEOM",
    num_normalization="minmax", optimization="none",  # nice_base: no refinement
)

x_f = X_te_sc[TE_IDX]
p_f = float(predict_fn(x_f[None, :])[0, 1])
yhat_f = int(p_f >= 0.5)
y_desired = 1 - yhat_f
print(f"factual: p_f={p_f:.6f}, yhat_f={yhat_f}, y_desired={y_desired}", flush=True)

res = nice_obj.explain(x_f[None, :])
x_cf = np.asarray(res).reshape(-1)
print(f"NICE returned a candidate (nearest justified neighbour), shape={x_cf.shape}", flush=True)

# ---- Find this exact row in NICE's own cached, build-time BATCHED prediction ----
train_X = nice_obj.data.X_train
matches = np.where((train_X == x_cf).all(axis=1))[0]
print(f"matching row(s) in X_train: {matches}", flush=True)
if len(matches):
    row = matches[0]
    cached_proba = nice_obj.data.train_proba[row]
    cached_class = int(nice_obj.data.X_train_class[row])
    print(f"NICE's CACHED batched prediction for this row (computed once, over the whole "
          f"training set, at build time): proba={cached_proba}, predicted class={cached_class} "
          f"(target was {y_desired}) -> {'MATCHES target' if cached_class == y_desired else 'DOES NOT MATCH target'}",
          flush=True)

# ---- Fresh re-checks: batch vs solo, float64 vs float32 ----
p_batch_fresh = predict_fn(X_tr_sc)[matches[0], 1] if len(matches) else None
p_solo_f64 = float(predict_fn(x_cf.astype(np.float64)[None, :])[0, 1])
p_solo_f32 = float(predict_fn(x_cf.astype(np.float32).astype(np.float64)[None, :])[0, 1])
# ^ cast to float32 then back to float64 for predict_fn's own dtype requirement,
#   mirroring nice_cf_one's np.asarray(x_cf, dtype=np.float32) precision loss.

print(f"\nFresh batched re-score (whole X_train in one call), this row's p(class=1): "
      f"{p_batch_fresh:.6f}" if p_batch_fresh is not None else "n/a", flush=True)
print(f"Solo re-score, float64, p(class=1): {p_solo_f64:.6f} -> yhat={int(p_solo_f64>=0.5)}", flush=True)
print(f"Solo re-score, float32-then-back,  p(class=1): {p_solo_f32:.6f} -> yhat={int(p_solo_f32>=0.5)}", flush=True)

yhat_solo_f64 = int(p_solo_f64 >= 0.5)
yhat_solo_f32 = int(p_solo_f32 >= 0.5)

print(f"\n===== VERDICT =====", flush=True)
print(f"y_desired = {y_desired}", flush=True)
if len(matches):
    print(f"Batched-at-build-time prediction: class {cached_class} "
          f"({'agrees' if cached_class==y_desired else 'DISAGREES'} with target)", flush=True)
print(f"Solo (float64) prediction:         class {yhat_solo_f64} "
      f"({'agrees' if yhat_solo_f64==y_desired else 'DISAGREES'} with target)", flush=True)
print(f"Solo (float32-cast) prediction:    class {yhat_solo_f32} "
      f"({'agrees' if yhat_solo_f32==y_desired else 'DISAGREES'} with target)", flush=True)

if len(matches) and cached_class == y_desired and yhat_solo_f64 != y_desired:
    print("\n==> CONFIRMED: batch-context effect alone (float64, same precision) flips the "
          "prediction. This is a TabPFN in-context-learning effect, not a precision artifact.", flush=True)
elif len(matches) and cached_class == y_desired and yhat_solo_f64 == y_desired and yhat_solo_f32 != y_desired:
    print("\n==> CONFIRMED: float32 precision cast alone flips the prediction (batch context "
          "was not the cause here).", flush=True)
elif len(matches) and cached_class != y_desired:
    print("\n==> The cached batched prediction itself already disagreed with the target class "
          "-- unexpected given justified_cf's own filter; needs a closer look.", flush=True)
else:
    print("\n==> This specific instance did not reproduce a batch/solo or precision "
          "discrepancy -- try a different (seed, idx) from the 36.", flush=True)
