import numpy as np
from typing import Sequence, Tuple, Mapping


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


class PiecewiseModel(Model):
    def __init__(
        self,
        num_bins: int,
        bounds: Mapping[str, Tuple[float, float]] = None,
        bin_edges: Mapping[str, Tuple[float, float]] = None,
        parameter_names: Sequence[str] = None,
    ):
        self.num_bins = num_bins
        self.bounds = bounds
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
            self.bounds = [(vmin, vmax) for bin in range(self.num_bins)]
        elif array_bounds:
            self.bounds = array_bounds
        else:
            raise ValueError("Need to provide either vmin/vmax OR array of tuple bounds.")