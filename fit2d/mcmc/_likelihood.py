import numpy as np
import time
from typing import Sequence, Callable, Mapping, Union

from .._galaxy import Galaxy, RingModel
from .._velocity_field_generator import create_2d_velocity_field
from ..models import Model

def dynesty_lnlike(lnlike_func, normalization_func, lnlike_args, ):
    return lambda cube: lnlike_func(normalization_func(cube), *lnlike_args)


def emcee_lnlike(params, emcee_version: float, lnlike_args: Union[Mapping, Sequence]):
    # wraps the lnlike function because emcee expects 
    # tuple of (lnlike, blob) returned
    if isinstance(lnlike_args, Sequence):
        lnl = lnlike(params, *lnlike_args), None
    else:
        lnl = lnlike(params, **lnlike_args)
    if emcee_version < 3:
        # older version expects two returns: lnlike and blobs
        return lnl, None
    else:
        return lnl


def chisq_2d(
    vlos_2d_model: np.ndarray,
    vlos_2d_obs: np.ndarray,
    v_err_2d: np.ndarray,
    v_err_const: float,
    regularization_coeff: float = 0.,
    return_n_pixels: bool=False,
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
        Chi^2 normalized by number of points
    """
    if np.shape(vlos_2d_model) != np.shape(vlos_2d_obs):
        raise ValueError(
            f"Shape of modeled velocity field {vlos_2d_model.shape} must "
            f"be equal to shape of observed velocity field {vlos_2d_obs.shape}."
        )
    if v_err_2d is not None:
        if np.shape(vlos_2d_obs) != np.shape(v_err_2d):
            raise ValueError(
                f"Shape of 2d error per pixel field {v_err_2d.shape} must "
                f"be equal to shape of observed velocity field {vlos_2d_obs.shape}."
            )
        chisq_array = (vlos_2d_obs - vlos_2d_model) ** 2 / v_err_2d ** 2
    elif v_err_const:
        chisq_array = (vlos_2d_obs - vlos_2d_model) ** 2 / v_err_const ** 2
    else:
        raise ValueError(
            "Must provide at least one of v_err_const (float) or "
            "v_err_2d (ndarray) to chisq_2d.")
    num_points = np.count_nonzero(~np.isnan(chisq_array))
    if num_points > 0 :
        chisq = np.nansum(chisq_array) + regularization_coeff * (-num_points)
        if return_n_pixels:
            return chisq, num_points
        else:
            return chisq
    else:
        raise ValueError(
            "There are no overlapping pixels between the modeled region and "
            "the first moment map.")       
    

def _tophat_prior(params: np.ndarray, bounds: Sequence[tuple]):
    if len(params) != len(bounds):
        raise ValueError(
            f"Length of params vector {len(params)} must be same as length of "
            f"model params {len(bounds)}."
    )
    bounds_min, bounds_max = bounds.T[0], bounds.T[1]
    if all(params > bounds_min) and all(params < bounds_max):
        return 0.
    else:
        return -np.inf


def lnlike(
    params: np.ndarray,
    model: Model,
    rotation_curve_func_kwargs: Mapping,
    galaxy: Galaxy,
    ring_model: RingModel,
    mask_sigma: float = 1.,
    v_err_2d: np.ndarray = None,
    v_err_const: float = None,
    n_interp_r: int = 150,
    n_interp_theta: int = 150,
    fit_structural_params: Mapping[str, int] = None,
    regularization_coeff: float = 0.,
    return_n_pixels: bool=False
):
    """[summary]

    Args:

        fit_structural_params: dict of structural ring parameter name (as
            specificied in the RingModel) and its corresponding index in
            params to be fit. e.g. if "inc

    Returns:
        [type]: [description]
    """

    if v_err_2d is None and v_err_const is None:
        raise ValueError(
            "Must provide at least one of v_err_const (float) or "
            "v_err_2d (ndarray) to lnlike.")
    elif v_err_2d is not None and v_err_const is not None:
        raise ValueError(
            "Only provide one of v_err_const (float) or "
            "v_err_2d (ndarray) to lnlike; you provided both.")
    params = np.array(params)
    if fit_structural_params:
        inc = params[fit_structural_params["inc"]]
        pos_angle = params[fit_structural_params["pos_angle"]]
        ring_model.update_structural_parameters(inc=inc, pos_angle=pos_angle)
    r_m, v_m = model.generate_1d_rotation_curve(params, **rotation_curve_func_kwargs)
    vlos_2d_model = create_2d_velocity_field(
        radii=r_m,
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
        regularization_coeff=regularization_coeff,
        return_n_pixels=return_n_pixels
    )
    prior = _tophat_prior(params, model.bounds)
    if return_n_pixels:
        return -0.5 * chisq[0] + prior, chisq[1]
    else:
        return -0.5 * chisq

