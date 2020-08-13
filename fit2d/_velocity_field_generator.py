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
        n_interp_r=75,
        n_interp_theta=700,
        n_neighbors_impute=2,
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

    for r in radii_interp:
        v = v_rot_interp(r)
        for theta in np.linspace(0, 2.0 * np.pi, n_interp_theta):
            # calculates the pixel coordinates and line of sight velocity in the observer frame
            x, y, v_los = _calc_v_los_at_r_theta(
                ring_model, v, r, theta, kpc_per_pixel, v_systemic
            )
            # checks that the pixel coords in observer frame are consistent
            # with image dimensions of first moment map
            if image_xdim - 1 > x > 0 and y < image_ydim - 1 and y > 0:
                arr_x, arr_y = int(np.round(x, 0)), int(np.round(y, 0))
                try:
                    v_field[arr_y][arr_x] = v_los
                except:
                    print(arr_x, arr_y, v_los)
    # fills in between the points that are sampled above. The mask is used so that we only
    # fill in pixels where the line of sight velocity is modeled within closest few pixels.
    near_neighbors_mask = create_blurred_mask(v_field)

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
    return (x_pix, y_pix)


def _calc_v_los_at_r_theta(ring_model, v_rot, r, theta, kpc_per_pixel, v_systemic):
    # transforms rotational velocity in galaxy frame to line of sight velocity in observer frame
    inc = ring_model.interp_ring_parameters["inc"](r)
    x0 = ring_model.interp_ring_parameters["x_center"](r)
    y0 = ring_model.interp_ring_parameters["y_center"](r)

    x_from_galaxy_center, y_from_galaxy_center = _convert_galaxy_to_observer_coords(
        ring_model, r, theta, kpc_per_pixel
    )
    v_los = v_rot * np.cos(theta) * np.sin(inc) + v_systemic
    x = x0 + x_from_galaxy_center
    y = y0 + y_from_galaxy_center

    return (x, y, v_los)