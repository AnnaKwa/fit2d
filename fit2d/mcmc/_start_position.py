import numpy as np
from typing import Sequence, Tuple, Mapping

from ._likelihood import lnlike

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


def nfw_start_points(
        nwalkers: int,
        bounds: Sequence[Tuple[float, float]],
        lnlike_args: Mapping,
        grid_points_per_param: Sequence[int] = None,
        random_seed: int = 1234,
):
    grid_points_per_param = grid_points_per_param or [4 for param in bounds]
    boundary_offsets = [0.1 * (bound[1] - bound[0]) for bound in bounds]

    start_grid = [
        np.linspace(bound[0] + boundary_offset, bound[1] - boundary_offset, num_to_sample)
        for bound, boundary_offset, num_to_sample
        in zip(bounds, boundary_offsets, grid_points_per_param)
        ]
    # explode to all possible combinations of starting params
    possible_start_combinations_0 = [row.flatten() for row in np.meshgrid(*start_grid)]
    possible_start_combinations = list(zip(*possible_start_combinations_0))

    lnlike_grid = [
        lnlike(
            parameter_space_point, **lnlike_args)[0]
        for parameter_space_point in possible_start_combinations]
    start_point = possible_start_combinations[np.argmax(np.array(lnlike_grid))]
    # random draw to start slightly away (5% of bounds range) from each start point
    start_point_radii = [0.05 * (bound[1]-bound[0]) for bound in bounds]
    return start_point, start_point_radii