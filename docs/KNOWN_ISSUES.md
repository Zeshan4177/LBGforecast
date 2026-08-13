# Known Issues

Found during a source-level documentation audit — not exhaustive.

1. **`compile_nzs.py` is currently non-functional**: `nruns = sys.argv[2]` is never cast to `int`, so `range(nruns)` raises `TypeError`; separately, the final `np.save(path+"...", )` call is missing its second (array) argument, which would also raise `TypeError` even if the first bug were fixed. Needs a fix before use.

2. **`priors_gp_dust.py`: the `*_nag` (Nagaraj+22) dust-sampling methods reference `self.n`, `self.tau`, `self.tau1`, `self.sfr` attributes that are not always set in `__init__`** — calling these methods can raise `AttributeError` depending on which code path populated the object. Worth auditing before relying on the Nagaraj-calibrated dust variant.

3. **Dead/superseded code**: `dust_priors.py` is imported by `priors_gp_dust.py` but never used there (dead import). `priors_gp.py` is a superseded legacy prior module with a hardcoded absolute path to a specific developer's machine (`/Users/fpetri/repos/LBGForecast/gp_models/`) — effectively unreachable from the production pipeline. `selection.py` is superseded by `colour_cuts.py` + `noise.py` (only referenced from older notebooks). `distributions.py` is not imported anywhere in the repo. `igm.py` duplicates most of `lyalpha.py`'s Lyman-break optical-depth calculation. `lbg_forecast/lsst_filters_old/` is a data directory never referenced by any code.

4. **`setup.py` is broken**: `package_dir={"lbg_forecast": "LBGforecast/lbg_forecast"}` and `package_data={"lbg_forecast": ["4pca_data/*.npy"]}` both point at paths that don't exist anywhere in the repo (leftover from an old repo restructuring) — `pip install -e .` currently fails. Install dependencies via `requirements.txt` and run scripts from the repo root instead, until this is fixed.

5. **Hardcoded machine-specific absolute paths**: `lbg_forecast/sps.py` (`SPS_HOME = "/Users/bl/software/fsps"`), `train_nn.py` (`SPS_HOME = "/Users/fpetri/packages/fsps"`), and `priors_gp.py` (multiple `/Users/fpetri/repos/LBGForecast/...` paths) all need manual editing per machine before running elsewhere.

6. **Stale notebooks that will fail on import**: `test_getmags.ipynb`, `test_goldrush.ipynb`, `test_lyalpha.ipynb`, `view_lsst_cuts.ipynb` import modules that no longer exist in the package (`lbg_forecast.hyperparameters`, `lbg_forecast.priors`, `lbg_forecast.priors_old`) — these reference a prior version of the codebase. `test_goldrush.ipynb` and `view_lsst_cuts.ipynb` also hardcode `/Users/fpetri/...` paths.

7. **`jax`/`jax-cosmo` undeclared dependency**: required by `angular_power.py`, `likelihood.py`, and the `modified_*.py` clustering/likelihood modules, but absent from both `requirements.txt` and `requirements_test.txt` — install manually or the forecast step will fail.

8. **`requirements.txt` vs `requirements_test.txt` disagree** on several version pins (TensorFlow, Keras, NumPy, SciPy) and only `requirements_test.txt` pins an exact Speculator commit — worth deciding which is authoritative and consolidating.

9. **Photometry emulator (Speculator/Photulator) is frozen**: per `EMULATOR-STATUS.md` at the repo root (dated 2026-07-11), this repo's Speculator/Photulator usage predates a 2026 convention migration in the upstream `speculator` package; the saved `trained_models/model_0x0lsst_*` weights' behavior under load depends on which class-definition convention they were trained under — don't revive/reuse without following the migration plan referenced in that file.

10. **`photo_to_nz.py`'s 3rd CLI argument is a no-op**: some call sites pass a third positional argument (e.g. the README used to show `photo_to_nz.py path id 0`), but the script hardcodes `extra=1` internally and never reads `sys.argv[3]` — the argument currently has no effect.

None of these block the core pipeline (steps 1-3, 5) from running — they affect the PCA compilation step, the legacy/frozen dust and emulator variants, and packaging, specifically.
