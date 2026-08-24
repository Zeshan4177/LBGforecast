# LBGForecast

Forecasts constraints on cosmological parameters from LSST Lyman-Break Galaxies (LBGs) at z~3-5. Incorporates redshift-distribution (n(z)) uncertainties using Stellar Population Synthesis (SPS) simulations.

# Background

Lyman-break galaxies (LBGs) are star-forming galaxies identified by the sharp drop in their observed flux blueward of the redshifted Lyman limit — a colour "dropout" between adjacent photometric bands, rather than a measured spectroscopic redshift. LSST will find enormous numbers of them at z ~ 3-5, a regime hard to reach spectroscopically at that depth, making them attractive tracers of large-scale structure for cosmology. The catch: a colour-selected sample's true redshifts are entangled with each galaxy's (unknown) dust attenuation, star-formation history and stellar mass, so the sample's redshift distribution, n(z), is itself uncertain — and that uncertainty, left unpropagated, biases and inflates the cosmological constraints drawn from the sample's angular clustering (Cl power spectra).

LBGForecast quantifies that effect end-to-end:

1. Sample galaxy properties (dust, star-formation history, mass, redshift, …) from priors calibrated to literature data as Gaussian Processes in redshift.
2. Simulate or emulate the resulting LSST photometry with Stellar Population Synthesis (SPS) modelling.
3. Apply realistic photometric noise and Lyman-break colour cuts to select simulated dropout samples, and derive their n(z).
4. Compress the resulting ensemble of n(z) realisations with PCA (Principal Component Analysis) into a handful of coefficients.
5. Forecast cosmological constraints with a Fisher-matrix likelihood (a fast, linearised estimate of parameter uncertainty, used in place of a full MCMC) that analytically marginalises over that PCA-compressed n(z) uncertainty — turning "how much we don't know about n(z)" directly into "how much that costs us in cosmological precision."

This work is tied to a PhD thesis — the commit history upstream includes entries like "thesis corrections" and "reviewer comments work."

# Repository layout

- `lbg_forecast/` — the core Python package: priors, the FSPS interface, the photometry emulator, the noise and selection model, PCA n(z) modelling, and the angular power spectrum / Fisher likelihood.
- top-level `*.py` / `*.sh` — pipeline driver scripts (see "Running the forecast pipeline" below).
- `gp_models/` — trained Gaussian-Process prior weights (mass function, dust, CSFRD), produced by the `GP_*.ipynb` notebooks.
- `sps_parameter_samples/`, `photo_samples/`, `nz_samples/`, `training_data/`, `trained_models/` — pipeline intermediate and output data. The large `.npy` arrays are gitignored and not distributed with the repo (see "Data availability" below).
- `dust_data/`, `csfr_data/`, `current_best_model/`, `sfr_emulator/`, `inoue14/`, `corrections/` — reference and calibration data consumed by the priors and IGM (intergalactic-medium absorption) modules.
- root `*.ipynb` — analysis and test notebooks (dozens of them); see `docs/MODULE_REFERENCE.md` for which are pipeline entry points versus dev-test scratch notebooks.
- `docs/` — detailed module reference and known-issues notes (added alongside this README).

# Installation

1. Install FSPS (Flexible Stellar Population Synthesis — the code used here to generate model galaxy spectra), via `python-fsps`. This wraps a *compiled* FSPS installation, not a plain `pip install` — you need a working FSPS build and the `SPS_HOME` environment variable pointing at it. Note that `lbg_forecast/sps.py` and `train_nn.py` currently hardcode a developer's local `SPS_HOME` path at import time; you will need to edit these for your own machine (ideally replacing the hardcoded path with `os.environ["SPS_HOME"]`). See `docs/KNOWN_ISSUES.md`.
2. Install the Speculator (Alsing et al. 2019) fork used here as the neural-network photometry emulator ("Photulator"):
```
pip install git+https://github.com/fpetri115/speculator.git
```
3. `git clone` this repository, then install the remaining Python dependencies:
```
pip install -r requirements.txt
```
   Two things worth flagging about this file:
   - **`jax` and `jax-cosmo` are required but not listed.** The angular-power-spectrum and forecast-likelihood modules (`lbg_forecast/angular_power.py`, `likelihood.py`, and the `modified_*.py` clustering modules) import them, but they appear in neither `requirements.txt` nor `requirements_test.txt`. Install them separately (`pip install jax jax-cosmo`), or step 5 of the pipeline below will fail on import.
   - Two PyPI package names differ from their import names: `astro-prospector` is imported as `prospect`, and `astro-sedpy` is imported as `sedpy`.
4. `pip install -e .` (via `setup.py`) does **not** currently work — `setup.py`'s `package_dir`/`package_data` point at stale paths left over from an earlier repo layout. For now, install dependencies from `requirements.txt` and run the scripts below directly from the repo root rather than installing the package. See `docs/KNOWN_ISSUES.md`.

# Calibrating the priors

Before step 1 of the pipeline, the three Gaussian-Process priors (stellar mass function, dust attenuation, cosmic star-formation-rate density) need to be fit and saved to `gp_models/*.pth`. This is done with the notebooks `GP_MASSFUNC.ipynb`, `GP_DUST.ipynb` (or `GP_DUST_NAG.ipynb` for the variant calibrated against Nagaraj et al. 2022), and `GP_CSFRD.ipynb`. Pre-fit weights are already included under `gp_models/`, so you only need to re-run these notebooks if you want to refit against updated literature data.

# Running the forecast pipeline

### 1. Sample SPS Parameters from Priors

```
mpiexec -n nproc python sample_sps_params.py ngals nrealisations run path mean dust_choice
```
Uses MPI (splits the sampling across many CPU cores in parallel, since each galaxy is drawn independently) to sample SPS parameters for `ngals` galaxies, giving a total of `nproc x nrealisations` realisations. SPS parameters are saved as `path/sps_parameter_samples/sps_{run}.npy` and `sparams_{run}.npy`. Set `mean=1` to sample the mean of the prior, or `mean=0` to draw `nproc x nrealisations` different stochastic prior realisations. `dust_choice` selects which calibration of the dust attenuation model to sample from: `0` = COSMOS, `1` = IRAC, `2` = Nagaraj et al. (2022). `path` must end with a trailing `/` and point at a directory containing `gp_models/` and `dust_data/` (e.g. the repo root) — both are required even when `mean=1`.

### 2. Simulate photometry

#### Option 1: Use the emulator (faster, GPU recommended)
```
python sample_photometry.py path run batch_size
```
Generates noiseless LSST ugriz photometry using the trained Photulator networks in `trained_models/model_0x0lsst_{u,g,r,i,z}` (requires TensorFlow). Output is saved as `path/photo_samples/photo_{run}.npy`.

#### Option 2: Use FSPS directly (exact, slower)
```
mpiexec -n nproc python simulate_sps.py run path filters
```
Generates noiseless photometry with either LSST ugrizy (`filters="lsst"`) or HSC grizy (`filters="suprimecam"`) filters. Output is saved as `path/photo_samples/sim_photo_{run}_{filters}.npy`.

### 3. Apply noise and selection

```
python photo_to_nz.py path run
```
Applies photometric noise (via `photerr`) and Goldrush-style colour-colour (dropout) box cuts to select u-, g-, and r-dropout LBG candidates, then derives n(z) samples for each dropout selection. Outputs land in `nz_samples/` (`nz_{run}.npy`, `n_detected_{run}.npy`) and `sps_parameter_samples/selected_sps_{run}.npy`. Some call sites pass a third positional argument to this script; the script does not currently read it (it hardcodes an internal `extra=1`), so it is a no-op — the call above with two arguments is the accurate usage.

Two shell wrappers chain these three steps: `sample_nzs.sh` runs the chain once via `mpiexec`, and `batch_run_nzs.sh` loops the same chain over a range of `run` values (covering both stochastic-prior and mean-prior realisations), deleting the large intermediate files after each iteration. There is also a standalone script, `sample_nzs.py`, that does emulated photometry and n(z) derivation in a single process — it is not wired into either shell wrapper.

### 4. PCA Approximation

n(z) realisations from multiple `run`s are meant to be concatenated by `compile_nzs.py`. **This script currently has two bugs that make it non-functional as written** — an argument (`nruns`) that is used without being cast to an integer, and a `np.save` call that is missing its array argument — so it needs a fix before use (see `docs/KNOWN_ISSUES.md`). In the meantime the concatenation can be done manually. Once you have a compiled n(z) ensemble, `PROCCESSING_NZs.ipynb` (or the `NzModel` class in `lbg_forecast/nz_model.py`) fits a PCA + Gaussian approximation in PCA-coefficient space, separately for each dropout sample, and writes `4pca_data/npca_*.npy` artifacts. These are consumed by the `u_dropout`, `g_dropout`, and `r_dropout` classes in `lbg_forecast/modified_redshift.py`.

### 5. Forecast Cosmological Constraints

The `Likelihood` class in `lbg_forecast/likelihood.py` loads the PCA n(z) artifacts, builds mock angular-clustering (Cl) data with `lbg_forecast/angular_power.py`, and computes a Fisher-matrix forecast that analytically marginalises over the PCA-coefficient covariance — propagating the n(z) uncertainty into the cosmological constraint, via `marginalised_log_likelihood` in `modified_likelihood.py`. This step is driven interactively rather than from a script: `FORECAST.ipynb` runs the full forecast, and `test_wilsonwhite.ipynb` runs a reduced two-parameter (σ₈, bias) version.

# Module reference

`docs/MODULE_REFERENCE.md` has a full table of every script and package module: what it does, its CLI arguments or public functions, and its inputs/outputs. Consult it before running an unfamiliar part of the pipeline.

# Data availability

The large sampled and simulated data arrays (`.npy` files under `sps_parameter_samples/`, `photo_samples/`, `nz_samples/`, `training_data/`, `dust_data/`, etc.) are excluded from version control via `.gitignore` and are not currently distributed with the repository. A fresh clone will not include them — you need to regenerate them by running the pipeline scripts above, or obtain them separately from the maintainer. This is the most common thing that will look "missing" after cloning.

# Status & known issues

`docs/KNOWN_ISSUES.md` lists known bugs, dead code, and stale notebooks found in an audit of this codebase; check it before debugging something that may already be a known issue.

The neural-network photometry emulator (Speculator/Photulator, `trained_models/model_0x0lsst_*`) is currently **frozen** pending a convention migration in the upstream `speculator` package — see `EMULATOR-STATUS.md` at the repo root before relying on it.

# License

MIT License — see `lbg_forecast/LICENSE`.
