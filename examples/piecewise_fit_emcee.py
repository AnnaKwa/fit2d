from fit2d import Galaxy, RingModel
from fit2d.mcmc import LinearPrior
from fit2d.mcmc import emcee_lnlike
from fit2d.models import PiecewiseModel

from astropy.io import fits
from datetime import datetime
import joblib
import emcee
from emcee import EnsembleSampler, moves


# GALAXY PARAMS
name = "UGC3974"
distance = 8000. # [kpc]
observed_2d_vel_field_fits_file = "/home/anna/Desktop/fit2d/data/UGC3974_1mom.fits"
deg_per_pixel=4.17e-4
v_systemic=270. 
ring_param_file = "/home/anna/Desktop/fit2d/data/UGC3974_ring_parameters.txt"
# x and y dims are switched in ds9 fits display versus np array shape
fits_ydim, fits_xdim = fits.open(observed_2d_vel_field_fits_file)[0].data.shape
num_bins = 10

# LIKELIHOOD PARAMS
v_err_const = 10.
mask_sigma=1.
random_seed = 1234

# MCMC PARAMS
batch_size = 2
mcmc_nwalkers = 20
mcmc_niter = 4
mcmc_ndim = num_bins
mcmc_nthreads = 4
# Try increasing stretch scale factor a. version must be >=3 for this to be used.
mcmc_moves = moves.StretchMove(a = 2)  
mcmc_version = float(emcee.__version__[0])


# Do not change anything below this.

galaxy = Galaxy(
    name=name,
    distance=distance,
    observed_2d_vel_field_fits_file=observed_2d_vel_field_fits_file,
    deg_per_pixel=deg_per_pixel,
    v_systemic=v_systemic, 
)

ring_model = RingModel(
    ring_param_file=ring_param_file,
    fits_xdim=fits_xdim,
    fits_ydim=fits_ydim,
    distance=distance
)
piecewise_model = PiecewiseModel(num_bins=num_bins)
piecewise_model.set_bounds(0, 200)
piecewise_model.set_bin_edges(rmin=ring_model.radii_kpc[0], rmax=ring_model.radii_kpc[-1])

prior = LinearPrior(bounds=piecewise_model.bounds)
prior_transform = prior.transform_from_unit_cube

start_positions = piecewise_start_points(mcmc_nwalkers, piecewise_model.bounds, random_seed=random_seed)


fit_inputs = {
    "piecewise_model": piecewise_model,
    "galaxy": galaxy,
    "ring_model": ring_model,
    "prior_transform": prior_transform
}
with open("fit_inputs.pkl", "wb") as f:
    joblib.dump(fit_inputs, f)

rotation_curve_func_kwargs = {
    "radii_to_interpolate": ring_model.radii_kpc,}

lnlike_args = [
    piecewise_model,
    rotation_curve_func_kwargs, 
    galaxy, 
    ring_model, 
    mask_sigma
]

lnlike_args = {
    "model": piecewise_model,
    "rotation_curve_func_kwargs": rotation_curve_func_kwargs, 
    "galaxy": galaxy, 
    "ring_model": ring_model, 
    "mask_sigma": mask_sigma,
    "v_err_const": v_err_const,
    }
    
sampler = EnsembleSampler(
    mcmc_nwalkers,
    mcmc_ndim, 
    emcee_lnlike, 
    args=[mcmc_version, lnlike_args], 
    threads=mcmc_nthreads,
)
if mcmc_version >= 3:
    sampler._moves = [mcmc_moves]


# this will break up the fitting procedure into smaller chunks of size batch_size and save progress
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%b-%Y")

mcmc_output = []
for batch in range(mcmc_niter // batch_size):
    print(f"Fitting batch {batch}")
    if batch == 0:
        batch_start = start_positions
    else:
        batch_start = None
        sampler.pool = temp_pool
    mcmc_output += sampler.run_mcmc(batch_start, batch_size)
    temp_pool = sampler.pool
    del sampler.pool
    with open(f'sampler_{timestampStr}.pkl', 'wb') as f:
        joblib.dump(sampler, f)
    print(f"Done with steps {batch*batch_size} - {(batch+1)*batch_size} out of {mcmc_niter}")

print(f"Done with emcee fit for {galaxy.name}")
