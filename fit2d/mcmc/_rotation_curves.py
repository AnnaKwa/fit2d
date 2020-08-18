import numpy as np
from typing import Sequence
from ..models import PiecewiseModel

"""
functions that produce 1D rotation curves from the MCMC parameters
first argument is always the parameters
"""

def piecewise_constant(
    params: Sequence[float],
    radii_to_interpolate: Sequence[float],
    piecewise_model: PiecewiseModel,
):
    vels = []
    velocities_at_piecewise_bin_centers = params
    bin_edges = piecewise_model.bin_edges
    for ring in radii_to_interpolate:
        for radius in range(len(bin_edges)):
            if (
                ring <= bin_edges[radius] and ring > bin_edges[radius - 1]
            ):  # ring is greater than current bin edge, and less than
                vels.append(
                    velocities_at_piecewise_bin_centers[radius - 1]
                )  # previous bin edge
    return np.array(vels)
