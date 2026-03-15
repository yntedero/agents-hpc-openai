"""TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)."""

import numpy as np


def topsis_rank(candidates, criteria, weights=None):
    """Rank candidates using TOPSIS.

    Parameters
    ----------
    candidates : list[dict]
        Each dict must contain keys listed in *criteria*.
    criteria : list[tuple[str, str]]
        Each entry is (column_name, "min" or "max").
        "min" = lower is better, "max" = higher is better.
    weights : list[float] | None
        Weight per criterion. Equal weights if None.

    Returns
    -------
    list[dict]
        Candidates sorted best-first, with added ``topsis_score`` key.
    """
    if not candidates:
        return []

    n = len(candidates)
    m = len(criteria)

    matrix = np.zeros((n, m))
    for i, c in enumerate(candidates):
        for j, (col, _) in enumerate(criteria):
            matrix[i, j] = float(c[col])

    # Normalize columns (vector normalization)
    norms = np.sqrt((matrix ** 2).sum(axis=0))
    norms[norms == 0] = 1.0
    norm_matrix = matrix / norms

    # Apply weights
    if weights is None:
        weights = np.ones(m) / m
    else:
        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()
    weighted = norm_matrix * weights

    # Ideal and anti-ideal solutions
    ideal = np.zeros(m)
    anti_ideal = np.zeros(m)
    for j, (_, direction) in enumerate(criteria):
        col = weighted[:, j]
        if direction == "min":
            ideal[j] = col.min()
            anti_ideal[j] = col.max()
        else:
            ideal[j] = col.max()
            anti_ideal[j] = col.min()

    # Distances
    dist_ideal = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    dist_anti = np.sqrt(((weighted - anti_ideal) ** 2).sum(axis=1))

    denom = dist_ideal + dist_anti
    denom[denom == 0] = 1.0
    scores = dist_anti / denom  # higher = better

    # Attach scores and sort
    result = []
    for i, c in enumerate(candidates):
        row = dict(c)
        row["topsis_score"] = round(float(scores[i]), 6)
        result.append(row)

    result.sort(key=lambda r: r["topsis_score"], reverse=True)
    return result
