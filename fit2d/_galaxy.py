from astropy.io import fits
from dataclasses import dataclass
from missingpy import KNNImputer
import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple, Sequence
import warnings

from ._constants import RAD_PER_ARCSEC
from ._helpers import (
    calc_physical_distance_per_pixel,
    _interpolate_baryonic_rotation_curve,
    create_blurred_mask
)

from skimage import color, data, restoration # added by stepen for deconvolution
from scipy.signal import fftconvolve as fftconv # faster convolution package - FastFourierTransform




@dataclass
class Galaxy:
    name: str
    distance: float
    observed_2d_vel_field_fits_file: str
    observed_2d_intensity_field_fits_file: str # added by Stephen
    deg_per_pixel: float
    v_systemic: float
    gas_radii: Sequence[float] = None
    gas_velocities: Sequence[float] = None
    stellar_radii: Sequence[float] = None
    stellar_velocities: Sequence[float] = None
    age: float = None
    observed_2d_dispersion_fits_file: str = None
    min_dispersion: float = 0.01

    """
    name: galaxy name
    distance: distance in kpc
    vlos_2d_data: np array of 1st moment, from the fits data
    deg_per_pixel: degrees per pixel in data array
    age: in Gyr, optional. Used for SIDM model.
    stellar/gas radii: Radii that define the stellar/gas rotation curve.
        If None, defaults to zero rotation velocity for this component.
    stellar/gas velocities: Stellar/gas rotation curve.
        If None, defaults to zero rotation velocity for this component.
    observed_2d_dispersion_fits_file: Velocity dispersion fits file. Optional.
    """

    def __post_init__(self):
        self.kpc_per_pixel = calc_physical_distance_per_pixel(
            self.distance, self.deg_per_pixel
        )
        for attribute in [
            "gas_radii",
            "gas_velocities",
            "stellar_radii",
            "stellar_velocities",
        ]:
            if getattr(self, attribute) is None:
                setattr(self, attribute, [0.0])
        observed_2d_vel_field = fits.open(self.observed_2d_vel_field_fits_file)[
            0
        ].data
        self.observed_2d_vel_field = self._correct_vel_field_units(observed_2d_vel_field)
        self.image_xdim, self.image_ydim = self.observed_2d_vel_field.shape
        if self.observed_2d_dispersion_fits_file:
            observed_2d_dispersion = fits.open(self.observed_2d_dispersion_fits_file)[0].data
            self.observed_2d_dispersion = self.impute_dispersion_map(observed_2d_dispersion, self.min_dispersion)
        self.observed_2d_intensity_field = fits.open(self.observed_2d_intensity_field_fits_file)[0].data # added by stephen
        # defining circular 2D Gaussian beam
        bmaj = fits.open(self.observed_2d_intensity_field_fits_file)[0].header["BMAJ"]/self.deg_per_pixel
        x, y = np.meshgrid(np.arange(0,bmaj), np.arange(0,bmaj))
        B = 2*np.pi*bmaj**2*np.exp(-1/(2*(bmaj**2)) * ((x-bmaj/2)**2 + (y-bmaj/2)**2))
        B /= np.sum(B)
        self.kernel = B

        data = np.nan_to_num(self.observed_2d_intensity_field, nan = 1e-9) # obs 0th mom w/ filled nans
        self.H1_map = restoration.richardson_lucy(data, self.kernel)

        self.m0 = fftconv(self.H1_map,self.kernel, mode = 'same') # m0


    @staticmethod
    def impute_dispersion_map(dispersion: np.ndarray, min_dispersion: float=0.01):
        dispersion[dispersion == 0.] = np.nan
        near_neighbors_mask = create_blurred_mask(dispersion, sigma=2.)
        imputer = KNNImputer(n_neighbors=3, weights="distance")
        dispersion = imputer.fit_transform(np.where(near_neighbors_mask == 1, dispersion, 0.0))
        dispersion[dispersion == 0] = np.nan
        return np.clip(dispersion, a_min=min_dispersion, a_max=None)

    @staticmethod
    def _correct_vel_field_units(v_field: np.ndarray):
        # Have seen instance where first moment map header units are km/s
        # but values are in m/s (of order 1e5)
        oom = np.nanmean(np.floor(np.log10(v_field)))
        if oom > 3:
            v_field = v_field / 1e3
        return v_field


class Constant:
    # workaround for pickle not being able to serialize lambda that
    # was used in
    # https://stackoverflow.com/a/12022055
    def __init__(self, constant):
        self.constant = constant
    def __call__(self, r):
        # replaces calls of lambda: x
        # the radius provided does not get used as this class returns
        # a constant inc/pa at any radius
        return self.constant


class RingModel:
    def __init__(
        self, ring_param_file: str, fits_xdim: int, fits_ydim: int, distance: float
    ):
        (
            self.radii_arcsec,
            self.bbarolo_fit_rotation_curve,
            inclinations,
            position_angles,
            fits_x_centers,
            fits_y_centers,
            self.v_systemics,
        ) = np.loadtxt(ring_param_file, usecols=(1, 3, 6, 5, -2, -1, 2)).T #I've changed the order of these when comparing against the 2D fit because the ringlog produced has a different ordering

        self.radii_kpc = self.radii_arcsec * RAD_PER_ARCSEC * distance
        self.inclinations = inclinations * np.pi / 180.
        self.position_angles = position_angles * np.pi / 180.

        _check_center_pixels(fits_x_centers, fits_y_centers, fits_xdim, fits_ydim)
        self.x_centers, self.y_centers = _convert_fits_to_array_coords(
            np.array(fits_x_centers), np.array(fits_y_centers), fits_xdim, fits_ydim
        )

        # interpolation functions for generating 2D velocity fields
        self.interp_ring_parameters = {
            "inc": interp1d(self.radii_kpc, self.inclinations),
            "pos_ang": interp1d(self.radii_kpc, self.position_angles),
            "x_center": interp1d(self.radii_kpc, self.x_centers),
            "y_center": interp1d(self.radii_kpc, self.y_centers),
        }


    def update_structural_parameters(self, inc=None, pos_angle=None):
        if inc:
            self.interp_ring_parameters["inc"] = Constant(inc)
        if pos_angle:
            self.interp_ring_parameters["pos_ang"] = Constant(pos_angle)



def _check_center_pixels(
    x_centers: Sequence[int], y_centers: Sequence[int], image_xdim: int, image_ydim: int
):
    for xc, yc in zip(x_centers, y_centers):
        if xc > image_xdim or yc > image_ydim:
            raise ValueError(
                "X, Y center pixel values lie outside the image "
                "dimensions. Check that they are within this range."
            )
        if (
            (0.25 > abs(1.0 * xc / image_xdim))
            or (abs(1.0 * xc / image_xdim) > 0.75)
            or (0.25 > abs(1.0 * yc / image_ydim))
            or (abs(1.0 * yc / image_ydim) > 0.75)
        ):
            warnings.warn(
                f"x, y  center pixel values {xc},{yc} provided are "
                f"not close to center of image dimensions "
                f"{image_xdim},{image_xdim}. "
                f"Is this intended?"
            )


def _convert_fits_to_array_coords(
    fits_x: np.ndarray, fits_y: np.ndarray, image_xdim: int, image_ydim: int
) -> Tuple[int, int]:
    # because x/y in ds9 fits viewer are different from x/y (row/col)
    # convention used here for numpy arrays
    array_y = image_ydim - fits_x
    array_x = image_xdim - fits_y
    return (array_x, array_y)
