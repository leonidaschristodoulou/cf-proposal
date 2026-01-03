from __future__ import annotations

import numpy as np

from typing import Dict, Any, Iterable, Optional, Tuple, List
from dataclasses import dataclass
from sklearn.datasets import make_classification

@dataclass(frozen=True)
class SynthDataset:
    """Container for one synthetic dataset."""
    name: str
    X: np.ndarray  # (n_samples, n_features), continuous floats
    y: np.ndarray  # (n_samples,), {0,1}
    meta: Dict[str, Any]


def generate_make_classification_suite(
    *,
    n_datasets: int = 20,
    n_samples: int = 2000,
    feature_grid: Iterable[int] = (5, 10, 20, 50, 100),
    redundancy_grid: Iterable[float] = (0.0, 0.25, 0.5, 0.75),
    informative_frac: float = 0.4,
    repeated_frac: float = 0.0,
    class_sep: float = 1.0,
    flip_y: float = 0.01,
    weights: Optional[Tuple[float, float]] = None,
    n_clusters_per_class: int = 2,
    random_state: int = 0,
    shuffle: bool = True,
    scale: bool = True,
) -> List[SynthDataset]:
    """
    Generate a suite of continuous (float) binary classification datasets with varying
    number of features and controlled redundancy using sklearn.make_classification.

    Key controls:
      - feature_grid: different dimensionalities
      - redundancy_grid: fraction of redundant features (linear combinations)

    Notes:
      - Redundancy makes feature attribution / sparse CF search harder because
        many directions are equivalent; expect your CF generator to either
        (a) change multiple correlated features, or (b) find brittle shortcuts.
      - For very high redundancy, see distance metrics behave weirdly
        unless use whitening / MAD scaling / covariance-aware distances.

    Returns:
      List of SynthDataset with metadata describing the generating parameters.
    """
    rng = np.random.default_rng(random_state)

    feature_grid = list(feature_grid)
    redundancy_grid = list(redundancy_grid)

    if n_datasets <= 0:
        return []

    out: List[SynthDataset] = []
    i = 0

    # We sample configs so you get a mix rather than a strict cartesian product.
    # But we keep it reproducible via rng.
    while len(out) < n_datasets:
        n_features = int(rng.choice(feature_grid))
        redundancy_frac = float(rng.choice(redundancy_grid))

        # Convert fractions -> counts, but keep the request valid for sklearn.
        n_informative = max(2, int(round(informative_frac * n_features)))
        n_repeated = int(round(repeated_frac * n_features))
        n_redundant = int(round(redundancy_frac * n_features))

        # Hard constraint: informative + redundant + repeated <= n_features
        # If violated, trim redundant first (most "optional"), then repeated.
        total = n_informative + n_redundant + n_repeated
        if total > n_features:
            overflow = total - n_features
            trim_red = min(overflow, n_redundant)
            n_redundant -= trim_red
            overflow -= trim_red
            if overflow > 0:
                trim_rep = min(overflow, n_repeated)
                n_repeated -= trim_rep
                overflow -= trim_rep
            # If still overflowing, reduce informative (rare unless informative_frac ~ 1)
            if overflow > 0:
                n_informative = max(2, n_informative - overflow)

        # A per-dataset seed so each dataset differs but is reproducible overall.
        ds_seed = int(rng.integers(0, 2**31 - 1))

        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=n_repeated,
            n_classes=2,
            n_clusters_per_class=n_clusters_per_class,
            weights=None if weights is None else list(weights),
            flip_y=flip_y,
            class_sep=class_sep,
            shuffle=shuffle,
            random_state=ds_seed,
        )

        # make_classification already returns continuous floats, but optionally standardize
        # so distances across datasets are more comparable.
        if scale:
            mu = X.mean(axis=0, keepdims=True)
            sig = X.std(axis=0, keepdims=True)
            sig[sig == 0.0] = 1.0
            X = (X - mu) / sig

        name = f"mc_f{n_features}_red{n_redundant}_inf{n_informative}_seed{ds_seed}"
        meta = dict(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=n_repeated,
            redundancy_frac=redundancy_frac,
            informative_frac=informative_frac,
            repeated_frac=repeated_frac,
            class_sep=class_sep,
            flip_y=flip_y,
            weights=weights,
            n_clusters_per_class=n_clusters_per_class,
            shuffle=shuffle,
            scaled=scale,
            seed=ds_seed,
        )
        out.append(SynthDataset(name=name, X=X.astype(np.float32), y=y.astype(np.int64), meta=meta))
        i += 1

    return out