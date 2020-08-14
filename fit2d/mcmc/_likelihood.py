import numpy as np
import time
from typing import Sequence

from .._galaxy import Galaxy, RingModel
from .._velocity_field_generator import create_2d_velocity_field
from ..models import PiecewiseModel


def dynesty_lnlike(lnlike_func, normalization_func, lnlike_args, ):
    return lambda cube: lnlike_func(normalization_func(cube), **lnlike_args)

def chisq_2d(
    vlos_2d_model: np.ndarray,
    vlos_2d_obs: np.ndarray,
    v_err_2d: np.ndarray = None,
    v_err_const: float = 10.0,
):
    """[summary]

    Args:

        vlos_2d_model (np.ndarray): modeled 2D velocity field generated by create_2d_velocity_field
        vlos_2d_obs (np.ndarray): observed2D velocity field
        v_err_2d (np.ndarray, optional): ndarray with values of 2D velocity uncertainty at each pixel.
            If used, should be of same dimensions as modeled velocity field. Defaults to None.
        v_err_const (float, optional): If no ndarray provided for errors at each pixel, use this
            constant value for calculating Chi^2. Defaults to 10 km/s.
        n_interp_r (int, optional): Number of radii to use in constructing modeled field.
            Defaults to 75.
        n_interp_theta (int, optional): Number of azimuthal angles to use in construction modeled field.
            Defaults to 700.

    Returns:
        Chi^2
    """
    if np.shape(vlos_2d_model) != np.shape(vlos_2d_obs):
        raise ValueError(
            f"Shape of modeled velocity field {vlos_2d_model.shape} must "
            f"be equal to shape of observed velocity field {vlos_2d_obs.shape}."
        )
    if v_err_2d:
        chisq = np.nansum((vlos_2d_obs - vlos_2d_model) ** 2 / v_err_2d ** 2)
    else:
        chisq = np.nansum((vlos_2d_obs - vlos_2d_model) ** 2 / v_err_const ** 2)
    return chisq


def lnlike_piecewise_model(
    params,
    galaxy: Galaxy,
    ring_model: RingModel,
    piecewise_model: PiecewiseModel,
    v_err_2d: np.ndarray = None,
    v_err_const: float = 10.0,
    n_interp_r: int = 150,
    n_interp_theta: int = 150,
    mask_sigma: float = 1.,
):

    v_m = _piecewise_constant(
        params, 
        radii_to_interpolate=ring_model.radii_kpc,
        piecewise_model=piecewise_model)
    if len(v_m) != len(ring_model.radii_kpc):
        raise ValueError(
            f"Number of radii returned by piecewise constant function ({len(v_m.shape)}) "
            f"must be equal to number of radii in Bbarolo parameter file({len(ring_model.radii_kpc)})."
        )
    vlos_2d_model = create_2d_velocity_field(
        radii=ring_model.radii_kpc,
        v_rot=v_m,
        ring_model=ring_model,
        kpc_per_pixel=galaxy.kpc_per_pixel,
        v_systemic=galaxy.v_systemic,
        image_xdim=galaxy.image_xdim,
        image_ydim=galaxy.image_ydim,
        n_interp_r=n_interp_r,
        n_interp_theta=n_interp_theta,
        mask_sigma=mask_sigma,
    )
    chisq = chisq_2d(
        vlos_2d_model=vlos_2d_model,
        vlos_2d_obs=galaxy.observed_2d_vel_field,
        v_err_2d=v_err_2d,
        v_err_const=v_err_const,
    )

    return -0.5 * (chisq)

def _piecewise_constant(
    velocities_at_piecewise_bin_centers: Sequence[float],
    radii_to_interpolate: Sequence[float],
    piecewise_model: PiecewiseModel,
):
    
    vels = []
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
