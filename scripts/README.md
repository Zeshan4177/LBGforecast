# `scripts/` — helpers for regenerating the n(z) ensemble

The emulator and the rest of the package cannot live in one environment: the saved
Photulator weights are TensorFlow pickles from the pre-2026 `speculator`, while the
`speculator` on most machines here is now the PyTorch rewrite. So the photometry step
runs in its own environment and hands `.npy` files back.

Build the emulator environment once:

    conda create -y -n lbgforecast_emu python=3.10
    conda activate lbgforecast_emu
    pip install "numpy==1.26.4" "tensorflow==2.16.2"
    pip install "git+https://github.com/fpetri115/speculator.git@b23bc1eccc91b80399450a3664182b152dafb64f"

Then, per batch of realisations:

    # 1. prior draws, in the main environment
    python scripts/sample_params.py . 50 100000 0 sps_parameter_samples/sps_new.npy

    # 2. emulated photometry, in the emulator environment
    conda run -n lbgforecast_emu python scripts/emulate_photometry.py . \
        sps_parameter_samples/sps_new.npy photo_samples/photo_new.npy

    # 3+4. noise, dropout cuts, n(z) and the PCA compression
    python build_nz_pca.py --runs new --npca 40

`dust_choice` in step 1 is 0 for the pop-cosmos dust prior, 2 for Nagaraj+22
(`dust_choice=1`, the IRAC variant, needs `dust_data/irac.txt`, which is not
distributed). A `dust_choice=2` batch is what the `_nag` artifacts need to stop
being placeholders.

## Measured cost, per 100,000 galaxies (this machine, 16 cores)

| step | time |
|---|---|
| 1. prior sampling | ~195 s, one core; embarrassingly parallel across realisations |
| 2. emulated photometry | ~12 s (8,500 galaxies/s, CPU) |
| 3. noise + dropout cuts | ~1.5 s |

Direct FSPS instead of the emulator is ~7.3 s **per galaxy**, i.e. 202 core-hours for
the same 100,000 — usable for validation, not for bulk generation.

## Emulator fidelity

Checked against the exact FSPS photometry in `photo_samples/sim_photo_0lsst.npy`
(20,000 galaxies): median offsets below 0.016 mag in every band, RMS 0.034-0.085 mag,
and within 20 < r < 26 (where the dropout cuts act) median offsets are below 0.006 mag.
That resolves the `EMULATOR-STATUS.md` GATE-1 question empirically — the saved weights
behave correctly under the pinned fork's forward pass.
