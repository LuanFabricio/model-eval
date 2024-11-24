from typing import List

import numpy as np


def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> List[float]:
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    cos_sim = np.dot(v1, v2)
    return 1 - cos_sim, cos_sim


def triplet_loss(
        anchor: np.ndarray,
        positive: np.ndarray,
        negative: np.ndarray,
        margin: float) -> float:

    ap_dist, _ = cosine_distance(anchor, positive)
    an_dist, _ = cosine_distance(anchor, negative)

    loss = ap_dist - an_dist + margin
    return max(loss, 0)
