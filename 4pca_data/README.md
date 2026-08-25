# `4pca_data/` — PCA compression of the LBG redshift distributions

These are the artifacts the forecast likelihood (step 5) consumes:
`modified_redshift.py` loads `npca_components_*`, `npca_mean_*` and `z_grid`;
`likelihood.py` loads `npca_means_*` and `npca_cov_*`.

## Provenance of the current files (regenerated 2026-08-25)

They were **not** recovered — they had never been committed (`.gitignore` excludes
`*.npy`) and no copy survives on any machine here. They were rebuilt from the SPS
parameter samples and exact FSPS photometry still on disk:

| input | realisations | galaxies each |
|---|---|---|
| `sps_parameter_samples/sps_0.npy` + `photo_samples/sim_photo_0lsst.npy` | 4 | 100,000 |
| `sps_parameter_samples/sps_1.npy` + `photo_samples/sim_photo_1lsst.npy` | 8 | 100,000 |

Regenerate with:

    python build_nz_pca.py --runs 0 1 --npca 10

## What is trustworthy, and what is not

The **machinery** is faithful: the same noise model (`photerr` LSST), the same
Goldrush dropout colour cuts, the same PCA + Gaussian approximation as the
published analysis. Section 5 runs end to end against these files and reproduces
the qualitative result — marginalising over n(z) uncertainty inflates sigma(sigma_8)
by ~15% and the LBG bias errors by 1.5-1.9x.

The **statistics** are not publication grade, for two reasons:

1. **Only 12 prior realisations.** The published analysis used 50 PCA components,
   which needs an ensemble of many hundreds. A centred ensemble of 12 has rank 11,
   so at most 10 components are usable before the coefficient covariance goes
   singular. The explained-variance spectrum is nearly flat (0.22, 0.17, 0.15, ...),
   which is the signature of an ensemble dominated by sampling noise rather than by
   a few genuine modes of prior uncertainty.
2. **Shot noise is a large fraction of the measured scatter.** Re-drawing only the
   photometric noise on a fixed realisation reproduces most of the realisation-to-
   realisation scatter: the prior-to-prior RMS is just 1.5-2.2x the noise-only RMS.
   For u-dropouts the spread in mean redshift is 0.015 across prior realisations
   against 0.014 from noise alone, i.e. the genuine prior variance is unresolved.
   Each realisation selects only ~1,800 u-, ~2,900 g- and ~480 r-dropouts from
   100,000 galaxies, so the histograms are Poisson limited.

Treat the current n(z) covariance as an order-of-magnitude placeholder that lets
the forecast code run, not as the uncertainty budget of the thesis.

## The `_nag` files are placeholders

`modified_redshift.py` loads the Nagaraj-dust (`_nag`) artifacts unconditionally, so
they must exist. No Nagaraj-dust (`dust_choice=2`) run survives, so they are copies
of the pop-cosmos ones. Consequence: any dust-model mismatch test (`mismatch_nag=`
in `Likelihood`) will show zero mismatch by construction. A genuine `_nag` ensemble
needs its own photometry pass; `build_nz_pca.py --genuine-nag` will then use it.

## Getting a publication-grade ensemble

The cost sits entirely in the photometry. Direct FSPS is **7.3 s per galaxy** on this
machine (202 core-hours per 100,000-galaxy realisation), so a few hundred realisations
is out of reach that way. The neural-network emulator exists for exactly this, but its
artifacts (`trained_models/model_0x0lsst_*.pkl`) are TensorFlow pickles from the
pre-2026 `speculator` and cannot be loaded by the current PyTorch `speculator` — see
`EMULATOR-STATUS.md`. Reviving it means installing the `fpetri115/speculator` fork in
an isolated environment, and validating its output against the FSPS photometry in
`photo_samples/` before trusting it.
