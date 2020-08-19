import numpy as np
from typing import Sequence, Tuple

# for using emcee, neet to generate starting positions


def piecewise_start_points(
        nwalkers: int,
        bounds: Sequence[Tuple[float, float]],
        random_seed: int = 1234,
) -> np.ndarray:
    start_points=[]
    nbins = len(bounds)
    np.random.seed(random_seed)
    for walker in range(nwalkers):
        start_pos = []
        for bin in range(nbins):
            start_pos.append(
                np.random.random()*(bounds[bin][1] - bounds[bin][0]) + bounds[bin][0])
        start_points.append(start_pos)
    return np.array(start_points)