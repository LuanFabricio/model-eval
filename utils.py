from typing import List

import numpy as np


def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> List[float]:
    v1_v2 = np.dot(v1, v2)
    v1_len = np.linalg.norm(v1)
    v2_len = np.linalg.norm(v2)

    cos_sim = v1_v2 / (v1_len * v2_len)
    return 1 - cos_sim, cos_sim
