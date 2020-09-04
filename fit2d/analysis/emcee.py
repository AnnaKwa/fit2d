import emcee
import glob
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Sequence


def get_output_files(galaxy_name, dir=None):
    if not dir:
        return sorted(glob.glob(f"sampler_{galaxy_name}*.pkl"))
    else:
        return sorted(glob.glob(os.path.join(dir, f"sampler_{galaxy_name}*.pkl")))
    

def combine_results_statistics(output_files,  min_iter=None, max_iter=None):
    # returns mean, std arrays
    mean, std = [], []
    for output_file in output_files:
        with open(output_file, "rb") as f:
            sampler = joblib.load(f)
        result = get_sampler_statistics(
            sampler, ["dummy_value"], min_iter, max_iter)["dummy_value"]
        mean.append(result["mean"])
        std.append(result["stddev"])
    return mean, std


def get_sampler_statistics(
    sampler ,
    param_names=None,
    min_iter = None,
    max_iter = None,
):
    _, _, nparams = sampler.chain.shape
    param_names = param_names or [f"param_{i}" for i in range(nparams)]
    assert len(param_names)==nparams, \
        "Number of param_names must equal number of parameters in sampler."
    
    iter_range = slice(min_iter, max_iter)
    
    params = {}
    for i, param_name in enumerate(param_names):
        mean = np.mean(sampler.chain[:, iter_range, i])
        stddev = np.std(sampler.chain[:, iter_range, i])
        params[param_name] = {
            "mean": mean,
            "stddev": stddev
        }
    return params


def plot_walker_paths(
        sampler: emcee.EnsembleSampler,
        param_names=None,
        min_iter = None,
        max_iter = None,
):
    nwalkers, niter, nparams = sampler.chain.shape
    param_names = param_names or [f"param_{i}" for i in range(nparams)]
    assert len(param_names)==nparams, \
        "Number of param_names must equal number of parameters in sampler."
    
    fig, ax = plt.subplots(nparams, sharex = True, figsize=(10,5*nparams))
    iter_range = slice(min_iter, max_iter)
    x = range(min_iter or 0, max_iter or niter)
    for i in range(nparams):
        for j in range(nwalkers):
            ax[i].plot(x, sampler.chain[j, iter_range, i])
        ax[i].set_ylabel(param_names[i])
        ax[i].set_xlabel('step')
    fig.subplots_adjust(hspace=0)
    plt.setp(
        [a.get_xtickparam_names() for a in fig.axes[:-1]], 
        visible=False)
    return fig


def plot_posterior_distributions(
        sampler: emcee.EnsembleSampler,
        param_names: Sequence[str] = None, 
        min_iter: int = None,
        max_iter: int = None
):
    nwalkers, _, nparams = sampler.chain.shape
    param_names = param_names or [f"param_{i}" for i in range(nparams)]
    assert len(param_names)==nparams, \
        "Number of param_names must equal number of parameters in sampler."
    fig = plt.figure(figsize=(5, 4*nparams))
    iter_range = slice(min_iter, max_iter)
    for i in range(nparams):
        ax = fig.add_subplot(nparams, 1, i+1)
        ax.hist(sampler.chain[:, iter_range, i].flatten())
        ax.set_xlabel(param_names[i])
    return fig