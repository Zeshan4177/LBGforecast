# Module Reference

Complete reference for every pipeline script and package module in this repository, for anyone who needs to modify or extend the code. The top-level README gives a high-level tour; this document is the detailed map.

## Pipeline scripts (repo root)

All paths are relative to a `path` CLI argument; from the repo root, `path="./"` works since `sps_parameter_samples/`, `photo_samples/`, `nz_samples/`, `gp_models/`, `dust_data/`, `csfr_data/`, `training_data/`, `trained_models/` all already exist there.

| Script | Purpose | CLI args (in order) | Reads | Writes | External deps |
|---|---|---|---|---|---|
| `sample_sps_params.py` | MPI-parallel sampler: draws per-galaxy SPS parameter vectors (redshift, mass, dust, CSFRD-linked SFH ratios, nuisance params) from the calibrated priors. | `ngals nrealisations run path mean` (`mean`: 1=prior mean only, 0=stochastic prior draws) | `gp_models/*.pth` (via `priors_gp_massfunc`, `priors_gp_dust`, `priors_gp_csfrd`) | `path/sps_parameter_samples/sps_{run}.npy` (shape `[nrealisations*nproc, ngals, 17]`), `sps_parameter_samples/sparams_{run}.npy` | MPI (`mpi4py`) |
| `sample_photometry.py` | Emulates noiseless LSST *ugriz* photometry for a saved SPS-parameter file using the trained Photulator/Speculator networks (fast, GPU-friendly alternative to running FSPS). | `path run batch_size` | `path/sps_parameter_samples/sps_{run}.npy`, `trained_models/model_0x0lsst_{u,g,r,i,z}` | `path/photo_samples/photo_{run}.npy` | TensorFlow, Speculator fork (GPU optional but recommended) |
| `simulate_sps.py` | MPI-parallel *exact* photometry simulation via real FSPS (not the emulator); used as ground truth / to generate emulator training data. | `run path filters` (`filters`: `"lsst"` (6 bands) or `"suprimecam"` (5 bands)) | `path/sps_parameter_samples/sps_{run}.npy` | `path/photo_samples/sim_photo_{run}{filters}.npy` | MPI, FSPS (`python-fsps`, needs `SPS_HOME`) |
| `photo_to_nz.py` | Applies photometric noise, LBG colour-colour selection (u/g/r-dropout cuts), and derives per-realisation n(z) samples + number densities. A 3rd CLI arg is accepted by some wrapper call sites but not read by the script (an internal `extra=1` is hardcoded). | `path run` | `path/sps_parameter_samples/sps_{run}.npy`, `photo_samples/photo_{run}.npy`, `sps_parameter_samples/sparams_{run}.npy` | `nz_samples/c_{run}.npy` (colours), `sps_parameter_samples/selected_sps_{run}.npy`, `nz_samples/nz_{run}.npy`, `nz_samples/n_detected_{run}.npy` | none beyond package |
| `compile_nzs.py` | Intended to concatenate multiple `nz_{run}_{i}.npy` chunk files into one compiled array. **Currently broken** — see `docs/KNOWN_ISSUES.md`. | `run nruns path` | `path/nz_samples/nz_{run}_{i}.npy` for `i` in `range(nruns)` | (intended) `nz_samples/nz_compiled_{run}.npy` | none |
| `sample_nzs.py` | Alternative single-process pipeline: emulates photometry *and* derives n(z) in one step (skips the separate `sample_photometry.py`/`photo_to_nz.py` split). Not invoked by either shell wrapper — a standalone/manual alternative. | `path batch_size run` | `path/sps_parameter_samples/sps_{run}.npy`, `trained_models/model_0x0lsst_*` | `path/nz_samples/nz_{run}.npy` | TensorFlow (via emulator) |
| `sample_nzs.sh` | Shell wrapper chaining the 3-stage production pipeline: `sample_sps_params.py` → `sample_photometry.py` → `photo_to_nz.py`. | `NPROC NGALS NREALS RUN DATA_PATH EXTRA MEAN BATCH_SIZE` | — | — | MPI (`mpiexec`) |
| `batch_run_nzs.sh` | Loops the same 3-stage pipeline `START..END` times, once with stochastic priors and once with mean-prior realisations per iteration, deleting large intermediate files after each iteration. | `NPROC NGALS NREALS RUN DATA_PATH BATCH START END` | — | `nz_samples/nz_{RUN}_{i}.npy`, `nz_samples/nz_{RUN}_mean_{i}.npy` (intermediates deleted) | MPI |
| `train_nn.py` | Trains the Photulator (Speculator) emulator, one filter band at a time. Hardcodes `SPS_HOME` for a specific developer machine — must be edited or changed to read the environment variable. | `select path file_id ndata load_model patience lr batch_size gradient_accumulation_steps add_final max_epochs validation_split` | `training_data/sps_parameters_{file_id}.npy`, `training_data/photometry_{file_id}.npy` | `trained_models/model_0x0{filter}`, `trained_models/loss_{filter}.npy`, `valloss_{filter}.npy` | TensorFlow/Keras, GPU recommended |
| `generate_training_data.py` | MPI script generating FSPS training data (SPS parameters + true photometry) for the emulator, using broad *uniform* priors rather than the calibrated GP priors. | `ngals path run_count` | — | `path/simulation_data/simulated_photometry_{rank+run_count}.npy`, `sps_parameters_{rank+run_count}.npy`, `spectra_*.npy`, `wavelengths_*.npy` | MPI, FSPS |

## `lbg_forecast/` package modules, by role

### Priors / population-parameter sampling

| Module | Purpose | Main public API |
|---|---|---|
| `population_model.py` | Core per-galaxy SPS-parameter sampler; orchestrates mass-function, dust and CSFRD priors into the 17-parameter vector used downstream. | `generate_sps_parameters()`, `truncated_normal()`, `modified_prospector_beta_sfh_prior()`, `prospector_beta_sfh_prior()`, `continuity_prior()`, `sps_parameter_names()` |
| `gaussian_priors.py` | Bounds + sampling for 6 nuisance Gaussian-prior parameters (`logzsol`, `igm_factor`, `gas_logu`, `gas_logz`, `log10fagn`, `agntau`). | `default_bounds()`, `sample_gaussian_prior_parameters()`, `gaussian_parameter_names()` |
| `priors_gp_massfunc.py` | Production stellar mass-function prior: Gaussian-process fits (in redshift) to literature Schechter-function parameters compiled from ~10 papers, with MCMC (`emcee`) sampling of `(z, log M)` pairs. | `MassFunctionPrior` class (`sample_logpdf`, `mass_function`, `lsst_number_density`, `n_z`) |
| `priors_gp_dust.py` | Production dust-attenuation prior (dust2/dust_index/dust1) vs. recent SFR, GP-calibrated against pop-cosmos samples. | `DustPrior` class (`sample_dust_model`, `sample_dust2/_index/_dust1`) |
| `priors_gp_csfrd.py` | Production cosmic star-formation-rate-density prior: GP fit to a literature CSFRD compilation, shifted relative to Behroozi et al. (2019) to correct a known observational systematic. | `CSFRDPrior` class (`sample_prior_corrected`, `get_prior_mean(_corrected)`) |
| `priors_mass_func.py` | Legacy/alternate redshift-mass prior (quadratic curves in z, smaller literature set); used only by `generate_training_data.py` for broad emulator-training coverage, not the production pipeline. | `preload_prior_data()`, `sample_redshift_mass_prior()` |
| `priors_gp.py` | Superseded legacy all-in-one GP prior module; hardcodes an absolute path to a specific developer's machine. Effectively dead code in the current pipeline (only reachable via an unused import in `dust_priors.py`). | `CSFRDPrior`, `DustIndexPrior`, `DiffuseDustPrior` |
| `dust_priors.py` | Alternative dust-prior implementations (sample from empirical data rather than GP-smoothed). Imported by `priors_gp_dust.py` but that import is unused there. | `DustPriorPop`, `DustPriorNag`, `DustPrior`, `sample_dust_model()` |
| `modified_prospector_beta.py` | Adapts `astro-prospector`'s "prospector-beta" (Wang et al. 2023) non-parametric SFH prior so its expected log-SFR-ratios track a sampled CSFRD curve. | `sample_logsfrratios()`, `get_csfrd_spline()` |
| `distributions.py` | Generic uniform/truncated-Gaussian sampling helpers. Not imported anywhere else in the repo (orphaned). | `sample_prior()`, `sample_prior_vec()`, `sample_hyperparams()` |

### SPS / spectral synthesis (FSPS interface)

| Module | Purpose | Main public API |
|---|---|---|
| `sps.py` | Core FSPS wrapper: builds a `fsps.StellarPopulation` model from the SPS parameter vector, generates the SED, redshifts it, integrates through LSST/HSC filters via `sedpy`. Hardcodes a local `SPS_HOME` path. | `initialise_sps_model()`, `update_model()`, `simulate_photometry()`, `get_magnitudes()` |
| `sfh.py` | Non-parametric "continuity prior" star-formation-history machinery, recent-SFR calculation, and alternate SFH parameterizations (Dirichlet, tau-model, double-power-law). | `default_agebins()`, `continuity_sfh()`, `logsfr_ratios_to_masses()`, `calculate_recent_sfr()` |
| `zhistory.py` | Chemical-enrichment (metallicity) history as a function of stellar-mass growth. | `sfr_to_zh()`, `sps_parameters_to_zhistory()` |
| `lyalpha.py` | Lyman-continuum + Lyman-series IGM optical-depth calculation plus Lyman-alpha line-peak perturbation, used inside `sps.get_magnitudes()`. | `lyman_continuum_tau()`, `apply_igm_attenuation()` |
| `igm.py` | A second, largely duplicate implementation of Lyman-continuum/series optical depth (near-identical to `lyalpha.py`); used from `sps.py`'s optional `modify_igm` path. | `lyman_continuum_tau()`, `apply_igm_attenuation()` |
| `cosmology.py` | Thin `astropy.cosmology` (WMAP1/WMAP9) wrapper, plus a lookup-table photometric correction between them. | `get_cosmology()`, `scale_to_z()`, `wmap1_to_9()` |

### Emulator

| Module | Purpose | Main public API |
|---|---|---|
| `emulator.py` | Loads the 5 trained Photulator networks (one per *ugriz* band) and does batched forward passes to emulate FSPS photometry from SPS parameters. | `fsps_emulator` class |

### Photometric noise, colour-cut selection, n(z)

| Module | Purpose | Main public API |
|---|---|---|
| `noise.py` | Applies LSST photometric noise via `photerr.LsstErrorModel`, then magnitude-based dropout pre-selection. | `setup_catalog()`, `get_noisy_magnitudes()` |
| `colour_cuts.py` | Production Lyman-break colour-colour (Goldrush-style) selection applied to noisy catalogs. | `select_dropouts()`, `apply_cuts_to_colours()` |
| `selection.py` | Superseded alternative selection (flat magnitude-limit cuts); only used from older notebooks, duplicates `colour_cuts.py`/`noise.py`. | `select_magnitudes()`, `colours()` |
| `nz.py` | Production per-realisation orchestrator: SPS params + photometry → noisy catalog → colour cuts → n(z) samples + number densities. | `calculate_nzs_from_photometry()`, `simulate_nzs()` |
| `nz_model.py` | PCA/Gaussian compression of simulated n(z) ensembles; produces the `4pca_data/*.npy` artifacts. Driven manually from `PROCCESSING_NZs.ipynb`/`test_pca.ipynb`, not called from any top-level script. | `NzModel` class, `perform_npca()`, `gauss_npca()` |

### Angular power spectrum / clustering forecast

(Adapted from the external `jax_cosmo` package, extended with this project's PCA-based n(z) and marginalised likelihood.)

| Module | Purpose | Main public API |
|---|---|---|
| `angular_power.py` | Theory and mock-data angular power spectra (Cl) for dropout clustering, with Gaussian covariance and noise. | `cl_theory()`, `cl_data()`, `pk()` |
| `modified_angular_cl.py` | Vendored/modified `jax_cosmo.angular_cl`: Limber-approximation Cl integration, noise Cl, Gaussian Cl covariance. | `angular_cl()`, `gaussian_cl_covariance()` |
| `modified_bias.py` | Vendored/modified `jax_cosmo` bias models, plus a project-specific interloper-vs-LBG bias split at a redshift cut. | `custom_bias`, `constant_linear_bias` |
| `modified_probes.py` | Vendored/modified `jax_cosmo.probes` (weak-lensing + number-counts kernels). | `WeakLensing`, `NumberCounts` |
| `modified_redshift.py` | Vendored/modified `jax_cosmo.redshift`, plus the project's 12-component PCA redshift-distribution classes reconstructing n(z) from `4pca_data/npca_*` artifacts. | `u_dropout`, `g_dropout`, `r_dropout` classes |
| `modified_likelihood.py` | Vendored/modified `jax_cosmo` Gaussian likelihood, plus `marginalised_log_likelihood()` which analytically marginalises over n(z) PCA-coefficient covariance. | `gaussian_log_likelihood()`, `marginalised_log_likelihood()` |
| `likelihood.py` | Top-level `Likelihood` class tying the above into a Fisher/likelihood object for cosmological-parameter forecasting; used from `FORECAST.ipynb`/`test_wilsonwhite.ipynb`. | `Likelihood` class (`.fisher()`, `.fisher_marg()`, `.logL()`) |

### Utilities

`utils.py` — sky-area constants, interloper-fraction helper, Fisher-ellipse contour plotting (`interlopers()`, `plot_contours()`).

## Notebooks (repo root, 39 total)

Entry points, roughly in pipeline order: `GP_MASSFUNC.ipynb`, `GP_DUST.ipynb`/`GP_DUST_NAG.ipynb`, `GP_CSFRD.ipynb` (fit the GP priors) → `PROCCESSING_NZs.ipynb`/`test_pca.ipynb` (PCA-compress n(z)) → `FORECAST.ipynb` (full cosmological forecast) → `test_wilsonwhite.ipynb` (reduced 2-parameter forecast).

Component/physics test notebooks (one module each, dev-test scratch): `test_cosmology`, `test_csfrd`, `test_dust_prior`, `test_massfunc`, `test_continuity_sfh`, `test_dirichlet_sfh`, `test_dynamic_sfh`, `test_sfhbins`, `test_igm_correction`, `tests_igm`, `test_lyalpha`, `test_modified_redshift`, `test_emulator`, `test_getmags`, `test_sps`, `test_prospector_beta`, `test_burstiness`, `test_sfr_emulator`, `test_noise_model`, `test_mlim`, `test_nz_generation`, `test_nz_generation_fast`.

Diagnostic/viewing notebooks: `view_lsst_cuts.ipynb`, `view_nz_sample_variance.ipynb`, `view_trained_models.ipynb`.

Note: several of the above are stale (see `docs/KNOWN_ISSUES.md`).

## External dependencies with a non-obvious install/import story

| Package | `requirements.txt` name | Import name | Note |
|---|---|---|---|
| FSPS | `fsps==0.4.7` | `fsps` | Wraps a compiled FSPS install; needs `SPS_HOME` env var. |
| Speculator/Photulator | not listed (only pinned in `requirements_test.txt`) | `speculator` | Installed separately per README; status frozen, see `EMULATOR-STATUS.md`. |
| Prospector | `astro-prospector==1.4.0` | `prospect` | PyPI name differs from import name. |
| sedpy | `astro-sedpy==0.3.2` | `sedpy` | PyPI name differs from import name. |
| DustE | `DustE==0.0.3` | `duste` | Used only in `priors_gp_dust.py`. |
| jax / jax-cosmo | **absent from both requirement files** | `jax`, `jax_cosmo` | Required for `angular_power.py`, `likelihood.py`, all `modified_*.py` — install manually. |
| MPI | `mpi4py==4.0.1` | `mpi4py` | Needs a system MPI implementation; run via `mpiexec`. |

---

Compiled from a source-level audit of the codebase; see docs/KNOWN_ISSUES.md for bugs and stale code found along the way.
