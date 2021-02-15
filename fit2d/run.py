import argparse
import logging
import yaml

from astropy.io import fits
import copy
from datetime import datetime
import glob
import joblib
import numpy as np
import os

import emcee
from emcee import EnsembleSampler, moves

from fit2d import Galaxy, RingModel
from fit2d.mcmc import LinearPrior
from fit2d.mcmc import emcee_lnlike, piecewise_start_points
from fit2d.models import PiecewiseModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fit2d")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=str, help="Name of configuration file with input args"
    )
    return parser.parse_args()

def mcmc(config, bin_edges, ring_model, ring_param_bounds, fit_structural_params):
    mcmc_version = float(emcee.__version__[0])
    stretch_move = config.get("mcmc_stretch_move_a", 2)
    mcmc_moves = moves.StretchMove(a = stretch_move)  

    for bin_index in range(config["num_bins"]):
        bin_min, bin_max = bin_edges[bin_index], bin_edges[bin_index+1]
        piecewise_model = PiecewiseModel(num_bins=1)
        piecewise_model.set_bounds(array_bounds=ring_param_bounds)
        piecewise_model.set_bin_edges(rmin=bin_min, rmax=bin_max)
        radii_to_interpolate = np.array([r for r in ring_model.radii_kpc if bin_min<r<bin_max])                              

        prior = LinearPrior(bounds=piecewise_model.bounds)
        start_positions = piecewise_start_points(config["mcmc_nwalkers"], piecewise_model.bounds, random_seed=config["random_seed"])  
        
        logger.info(f"Fitting ring {bin_index}")
        
        rotation_curve_func_kwargs = {
            "radii_to_interpolate": radii_to_interpolate}

        lnlike_args = {
            "model": piecewise_model,
            "rotation_curve_func_kwargs": rotation_curve_func_kwargs, 
            "galaxy": galaxy, 
            "ring_model": ring_model, 
            "mask_sigma": config["mask_sigma"],
            "v_err_const": config["v_err_const"],
            "v_err_2d": config["v_err_2d"],
            "fit_structural_params": fit_structural_params
            }

        sampler = EnsembleSampler(
            config["mcmc_nwalkers"],
            config["mcmc_ndim"], 
            emcee_lnlike, 
            args=[mcmc_version, lnlike_args], 
            threads=config.get("mcmc_nthreads", 1),
        )

        if mcmc_version >= 3:
            sampler._moves = [mcmc_moves]
        sampler_output_file = os.path.join(
            config.get("save_dir") or "",
            f"sampler_{galaxy.name}_ring{bin_index}_{bin_min:.2f}-{bin_max:.2f}.pkl")

        for batch in range(config["mcmc_niter"] // config["batch_size"]):
            if batch == 0:
                batch_start = start_positions
            else:
                batch_start = None
                sampler.pool = temp_pool
            sampler.run_mcmc(batch_start, config["batch_size"])
            temp_pool = sampler.pool
            del sampler.pool
            with open(sampler_output_file, 'wb') as f:
                sampler_copy = copy.copy(sampler)
                del sampler_copy.log_prob_fn
                joblib.dump(sampler_copy, f)
            logger.info(f'Done with steps {batch*config["batch_size"]} - {(batch+1)*config["batch_size"]} out of {config["mcmc_niter"]}')

    logger.info(f"Done with emcee fit for {galaxy.name}")


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f)
    
    # x and y dims are switched in ds9 fits display versus np array shape
    fits_ydim, fits_xdim = fits.open(config["observed_2d_vel_field_fits_file"])[0].data.shape

    mcmc_version = float(emcee.__version__[0])

    fit_structural_params = config.get("fit_structural_params", {})

    galaxy = Galaxy(
        name=config["name"],
        distance=config["distance"],
        observed_2d_vel_field_fits_file=config["observed_2d_vel_field_fits_file"],
        deg_per_pixel=config["deg_per_pixel"],
        v_systemic=config["v_systemic"], 
        observed_2d_dispersion_fits_file=config.get("observed_2d_dispersion_fits_file", None)
    )

    ring_model = RingModel(
        ring_param_file=config["ring_param_file"],
        fits_xdim=fits_xdim,
        fits_ydim=fits_ydim,
        distance=config["distance"]
    )
    ring_param_bounds = [
        (config["vmin"], config["vmax"]),
        (config["inc_min"], config["inc_max"]),
        (config["pos_angle_min"], config["pos_angle_max"])
    ]


    if "rmin" in config and "rmax" in config:
        bin_edges = np.linspace(config["rmin"], config["rmax"], config["num_bins"]+1)
    else:
        bin_edges = np.linspace(ring_model.radii_kpc[0], ring_model.radii_kpc[-1], config["num_bins"]+1)
    bin_centers = [(bin_edges[i]+bin_edges[i+1])/2. for i in range(config["num_bins"])]
    mcmc(config, bin_edges, ring_model, ring_param_bounds, fit_structural_params)