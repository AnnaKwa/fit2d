import numpy as np
from typing import Sequence, Tuple, Mapping
from .._constants import GNEWTON


class Model:
    def __init__(
        self,
        bounds: Mapping[str, Tuple[float, float]] = None,
        parameter_names: Sequence[str] = None,
    ):
        """[summary]

        Args:
            parameters (Sequence[str]): Ordered parameter names
            bounds (Mapping[str, Tuple[float, float]], optional): 
                Bounds for parameters. If not provided, will be (-inf, inf)
            log_parameters (Sequence[str], optional): Parameters to convert to log
                space when fitting MCMC.
        """

        self.parameter_names = parameter_names
        self.bounds = bounds
    
    def generate_1d_rotation_curve(self, params: Sequence[float], radii: np.ndarray):
        # the model should contain the information needed to calculate
        # the rotation curve with only the parameters and radii
        # passed to this method
        raise NotImplementedError


class NFWModel(Model):
    def __init__(
        self,
        bounds: Mapping[str, Tuple[float, float]],
        v_stellar: np.ndarray,
        v_gas: np.ndarray,
    ):
        self.bounds = bounds
        for param_bounds in bounds:
            if max(param_bounds) > 100:
                raise ValueError(
                    "Bounds for rhos and rs should be given as log values.")
        self.v_stellar = v_stellar
        self.v_gas = v_gas
        self.parameter_order = sorted(list(bounds.keys()))
        
    def generate_1d_rotation_curve(self, params: Sequence[float], radii: np.ndarray):
        # assumes that rhos and rs are passed as log params, and are
        # converted to physical parameters before using to calculate velocity
        ml = params[self.parameter_order.index("ml")]
        rhos = 10 ** params[self.parameter_order.index("rhos")]
        rs = 10 ** params[self.parameter_order.index("rs")]

        v2_baryons = ml * (self.v_stellar ** 2) + self.v_gas **2
        dm_mass_enclosed = self._nfw_mass_enclosed(radii, rhos, rs)
        v2_dm = GNEWTON * dm_mass_enclosed / radii
        return np.sqrt(v2_dm + v2_baryons)

    def _nfw_mass_enclosed(
            self,
            radii: np.ndarray,
            rho_s: float,  # msun kpc-3
            r_s: float # kpc
    ):
        mass_enclosed = 4. * np.pi * rho_s * r_s**3 * \
                        (np.log((r_s + radii) / r_s) -
                            (radii/ (radii + r_s) ))
        return mass_enclosed


class PiecewiseModel(Model):
    def __init__(
        self,
        num_bins: int,
        bounds: Sequence[Tuple[float, float]] = None,
        bin_edges: Mapping[str, Tuple[float, float]] = None,
        parameter_names: Sequence[str] = None,
    ):
        self.num_bins = num_bins
        self.bounds = np.array(bounds)
        self.bin_edges = bin_edges
        self.parameter_names = parameter_names

    def set_bin_edges(self, rmin: float, rmax: float):
        self.bin_edges = np.linspace(0.999 * rmin, 1.001 * rmax, self.num_bins+1)

    def set_bounds(
        self,
        vmin: float = None,
        vmax: float = None,
        array_bounds: Sequence[tuple] = None,
    ):
        """If vmin and vmax provided, all bins will use those as min/max bounds. An
        array array_bounds provided as a sequence of (min, max) for each bin. Only one
        option should be provided.

        Args:
            vmin (float, optional): Min to set for all bins. Defaults to None.
            vmax (float, optional): Max to set for all bins. Defaults to None.
            array_bounds (Sequence[tuple], optional): Custon bounds for each bin. Defaults to None.

        Raises:
            ValueError: [description]
        """
        if (vmin and vmax) and array_bounds:
            raise ValueError(
                "Both vmin/vmax and array_bounds were provided. "
                "Either provide vmin, vmax values to be used in all bins, "
                "or an array of tuples array_bounds, but not both."
            )
        if (vmin is not None and vmax is not None):
            self.bounds = np.array([(vmin, vmax) for bin in range(self.num_bins)])
        elif array_bounds is not None:
            self.bounds = np.array(array_bounds)
        else:
            raise ValueError("Need to provide either vmin/vmax OR array of tuple bounds.")

    def generate_1d_rotation_curve(
            self,
            params: Sequence[float],
            radii_to_interpolate: Sequence[float],
    ):
        vels = []
        velocities_at_piecewise_bin_centers = params
        bin_edges = self.bin_edges
        for ring in radii_to_interpolate:
            for radius in range(len(bin_edges)):
                if (
                    ring <= bin_edges[radius] and ring > bin_edges[radius - 1]
                ):  # ring is greater than current bin edge, and less than
                    vels.append(
                        velocities_at_piecewise_bin_centers[radius - 1]
                    )  # previous bin edge
        return radii_to_interpolate, np.array(vels)

