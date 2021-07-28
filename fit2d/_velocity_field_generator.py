from missingpy import KNNImputer
import numpy as np
from scipy.interpolate import interp1d
from typing import Sequence

from ._galaxy import RingModel
from ._helpers import create_blurred_mask

"""
This code takes a (r, theta) coordinate in the galaxy's coordinate frame
and the rotational velocity at that point (also in the galaxy's rest coordinate frame)
and calculates 
	i) the pixel position that corresponds to that coordinate, in the observer's frame,
  	accounting for position angle and inclination of the galaxy disk & systemic velocity.
  ii) the line of sight velocity (i.e. what the first moment map shows) at that pixel

We start in the galaxy coordinates and transform to the observer pixel coordinates, instead of vice versa,
because there is a one to one mapping of galaxy coord -> pixel but there is not a one to one mapping in the other direction
(multiple radii of the galaxy may lie in one pixel).

It iterates over many radii and angles in the galaxy frame and fills in the corresponding pixels.
Even then, however, there will be pixels that are not filled in with a line of sight velocity,
because we are sampling discrete points in the galaxy coordinate frame.

Therefore the last step is to use a nearest neighbors algorithm, which fill in the line of sight values for
empty pixels by interpolating the values of nearby pixels.
"""

def create_2d_velocity_field(
    radii: Sequence[float],
    v_rot: Sequence[float],
    ring_model: RingModel,
    kpc_per_pixel: float,
    v_systemic: float,
    image_xdim: int,
    image_ydim: int,
    n_interp_r=150,
    n_interp_theta=150,
    n_neighbors_impute=2,
    mask_sigma=1.,
    harmonic_coefficients=None
):
    """
        radii (Sequence[float]): radii for which modeled 1D velocities are provided. [kpc]
        v_rot (Sequence[float]): Modeled 1D velocities at radii.
        n_interp_r (int, optional): Number of radii to use in constructing modeled field.
            Defaults to 75.
        n_interp_theta (int, optional): Number of azimuthal angles to use in construction modeled field.
            Defaults to 700.
    """

    """
    uses tilted ring model parameters to calculate velocity field
    using eqn 1-3 of 1709.02049 and v_rot from mass model
    it is easier to loop through polar coordinates and then map the v_los to the
    nearest x,y point
    returns 2d velocity field array
    """
    # ndarray x/y dims are flipped from ds9 display
    v_field = np.zeros(shape=(image_ydim, image_xdim))
    v_field[:] = np.nan
    v_rot_interp = interp1d(radii, v_rot)

    radii_interp = np.linspace(np.min(radii), np.max(radii), n_interp_r)
    theta = np.linspace(0, 2.0 * np.pi, n_interp_theta)
    flattened_r_v_pairs = np.array(np.meshgrid(radii_interp, theta)).T.reshape(-1, 2).T
    r, theta = flattened_r_v_pairs[0], flattened_r_v_pairs[1]
    v = v_rot_interp(r)
    x, y, v_los = _calc_v_los_at_r_theta(
                ring_model, v, r, theta, kpc_per_pixel, v_systemic, harmonic_coefficients
            )
    x = np.round(x).astype(int) 
    y = np.round(y).astype(int)
    if y < image_ydim and x < image_xdim:
        v_field[y, x] = v_los

    near_neighbors_mask = create_blurred_mask(v_field, mask_sigma)

    imputer = KNNImputer(n_neighbors=n_neighbors_impute, weights="distance")
    v_field = imputer.fit_transform(np.where(near_neighbors_mask == 1, v_field, 0.0))
    v_field[v_field == 0] = np.nan

    # rotate to match the fits data field
    v_field = np.rot90(v_field, 3)
    return v_field


def _convert_galaxy_to_observer_coords(ring_model, r, theta, kpc_per_pixel):
    """
    Transforms the r, theta coords in galaxy frame to the x,y pixel coords of observer frame.

    :param r: physical distance from center [kpc]
    :param theta: azimuthal measured CCW from major axis in plane of disk
    :return: x, y coords in observer frame after applying inc and position angle adjustment
    """
    inc = ring_model.interp_ring_parameters["inc"](r)
    pos_ang = ring_model.interp_ring_parameters["pos_ang"](r)
    x_kpc = -r * (
            -np.cos(pos_ang) * np.cos(theta) + np.sin(pos_ang) * np.sin(theta) * np.cos(
        inc)
    )
    y_kpc = r * (
            np.sin(pos_ang) * np.cos(theta) + np.cos(pos_ang) * np.sin(theta) * np.cos(
        inc)
    )
    x_pix = x_kpc / kpc_per_pixel
    y_pix = y_kpc / kpc_per_pixel
    return x_pix, y_pix


def _harmonic_expansion_los_velocity(theta, harmonic_coefficients=None):
    # sum over n: c_n * cos(n * phi + phase_n) + s_n * sin(n * phi + phase_n)
    # ANGLES ARE IN RADIANS
    harmonic_coefficients = harmonic_coefficients or {
        "c1": 0., "s1": 0., "phase1": 0.,
        "c2": 0., "s2": 0., "phase2": 0.
        }
    fit_order = max([int(key[-1]) for key in harmonic_coefficients])
    sum = 0.
    for i in range(fit_order):
        n = i+1
        c_n = harmonic_coefficients.get(f"c{n}", 0.)
        s_n = harmonic_coefficients.get(f"s{n}", 0.)
        phase_n = harmonic_coefficients.get(f"phase{n}", 0)
        sum += c_n * np.cos(theta + phase_n) + s_n * np.sin(theta + phase_n)
    return sum


def _calc_v_los_at_r_theta(ring_model, v_rot, r, theta, kpc_per_pixel, v_systemic, harmonic_coefficients=None):
    # transforms rotational velocity in galaxy frame to line of sight velocity in observer frame
    inc = ring_model.interp_ring_parameters["inc"](r)
    x0 = ring_model.interp_ring_parameters["x_center"](r)
    y0 = ring_model.interp_ring_parameters["y_center"](r)

    x_from_galaxy_center, y_from_galaxy_center = _convert_galaxy_to_observer_coords(
        ring_model, r, theta, kpc_per_pixel
    )
    v_los = v_rot * np.cos(theta) * np.sin(inc) + v_systemic \
        + _harmonic_expansion_los_velocity(theta, harmonic_coefficients)

    x = x0 + x_from_galaxy_center
    y = y0 + y_from_galaxy_center

    return x, y, v_los