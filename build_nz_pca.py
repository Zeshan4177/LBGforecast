"""Derive the n(z) ensemble and its PCA compression from photometry already on disk.

This covers steps 3 and 4 of the pipeline in one pass, starting from SPS parameter
samples and *noiseless* photometry that have already been simulated:

    sps_parameter_samples/sps_{run}.npy      (nrealisations, ngals, 17)
    photo_samples/sim_photo_{run}lsst.npy    (nrealisations * ngals, 6)   [FSPS]
    photo_samples/photo_{run}.npy            (nrealisations, ngals, 5)    [emulator]

For each prior realisation it applies the photometric noise model and the Goldrush
dropout colour cuts, histograms the selected redshifts onto a common grid, and fits
the PCA + Gaussian approximation that the forecast likelihood consumes.

    python build_nz_pca.py --runs 0 1 --npca 10

Outputs land in redshifts/ (the n(z) ensemble) and 4pca_data/ (the PCA artifacts).
"""
import argparse
import os
import shutil
import time

import numpy as np

import lbg_forecast.nz as nz
from lbg_forecast.nz_model import NzModel

DZ = 0.1
ZMIN = 0.0
ZMAX = 7.0


def load_run(path, run):
    """Return (sps_params, noiseless_ugriz) for one run, whichever way it was simulated."""
    sps = np.asarray(np.load(os.path.join(path, "sps_parameter_samples", f"sps_{run}.npy")))
    nreal, ngals, _ = sps.shape

    fsps_file = os.path.join(path, "photo_samples", f"sim_photo_{run}lsst.npy")
    emu_file = os.path.join(path, "photo_samples", f"photo_{run}.npy")
    if os.path.exists(fsps_file):
        photo = np.asarray(np.load(fsps_file)).reshape(nreal, ngals, -1)
    elif os.path.exists(emu_file):
        photo = np.asarray(np.load(emu_file))
    else:
        raise FileNotFoundError(f"no photometry for run {run}: expected {fsps_file} or {emu_file}")

    # the selection model works in LSST ugriz; FSPS output also carries a y band
    return sps, photo[:, :, :5]


def derive_redshifts(path, runs):
    """Apply noise + dropout cuts to every realisation, returning selected redshifts."""
    per_run = []
    for run in runs:
        sps, photo = load_run(path, run)
        print(f"run {run}: {sps.shape[0]} realisations x {sps.shape[1]} galaxies", flush=True)
        out = np.empty((sps.shape[0], 3), dtype=object)
        for n in range(sps.shape[0]):
            t0 = time.time()
            nzs = nz.calculate_nzs_from_photometry(sps[n], photo[n], extra=False)
            for d in range(3):
                out[n, d] = np.asarray(nzs[d])
            print("   realisation %d: u=%5d g=%5d r=%5d selected  (%.1f s)"
                  % (n, len(nzs[0]), len(nzs[1]), len(nzs[2]), time.time() - t0), flush=True)
        np.save(os.path.join(path, "redshifts", f"emulated_redshifts_{run}.npy"), out, allow_pickle=True)
        per_run.append(out)
    return np.vstack(per_run)


def histogram_ensemble(raw):
    """Turn per-realisation redshift samples into normalised n(z) on a common grid."""
    bins = np.arange(ZMIN, ZMAX + DZ, DZ)
    z_grid = (bins[1:] + bins[:-1]) / 2
    ensemble = [np.array([np.histogram(raw[n, d], density=True, bins=bins)[0]
                          for n in range(raw.shape[0])]) for d in range(3)]
    return z_grid, ensemble


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", default=".", help="repository root (default: cwd)")
    parser.add_argument("--runs", nargs="+", default=["0", "1"], help="run labels to combine")
    parser.add_argument("--npca", type=int, default=None,
                        help="number of PCA components (default: as many as the ensemble supports)")
    parser.add_argument("--genuine-nag", action="store_true",
                        help="redshifts/nz*_nag.npy hold a real Nagaraj-dust (dust_choice=2) "
                             "ensemble rather than placeholders")
    args = parser.parse_args()

    path = os.path.abspath(args.path)
    os.makedirs(os.path.join(path, "redshifts"), exist_ok=True)
    os.makedirs(os.path.join(path, "4pca_data"), exist_ok=True)

    raw = derive_redshifts(path, args.runs)
    np.save(os.path.join(path, "redshifts", "emulated_redshifts_all.npy"), raw, allow_pickle=True)
    nreal = raw.shape[0]
    print("total realisations:", nreal)

    z_grid, ensemble = histogram_ensemble(raw)
    for arr, name in zip(ensemble, ["nzus", "nzgs", "nzrs"]):
        np.save(os.path.join(path, "redshifts", f"{name}.npy"), arr)
    np.save(os.path.join(path, "redshifts", "z_grid.npy"), z_grid)

    # A centred ensemble of N realisations has rank N-1, so the PCA coefficient
    # covariance goes singular if we ask for every component. Leave headroom.
    npca = args.npca if args.npca is not None else max(1, min(nreal - 2, len(z_grid)))
    if npca > nreal - 2:
        print(f"WARNING: {npca} components requested from only {nreal} realisations; "
              "the coefficient covariance may be singular or badly conditioned.")

    m = NzModel(path=path + os.sep, nag=False)
    m.save_npca_data(npca, path)
    print(f"wrote 4pca_data/ with {npca} components from {nreal} realisations")

    # modified_redshift.py loads the Nagaraj-dust (_nag) artifacts unconditionally,
    # so they have to exist even if no Nagaraj-dust run has been simulated. The
    # Nagaraj dust prior is deterministic given the recent star-formation rate, so a
    # genuine _nag ensemble needs its own photometry pass with dust_choice=2.
    if args.genuine_nag:
        m_nag = NzModel(path=path + os.sep, nag=True)
        m_nag.save_npca_data(npca, path)
        print("wrote the Nagaraj-dust (_nag) artifacts from redshifts/nz*_nag.npy")
    else:
        for name in ("nzus", "nzgs", "nzrs"):
            shutil.copyfile(os.path.join(path, "redshifts", f"{name}.npy"),
                            os.path.join(path, "redshifts", f"{name}_nag.npy"))
        m_nag = NzModel(path=path + os.sep, nag=True)
        m_nag.save_npca_data(npca, path)
        print("NOTE: the _nag artifacts are copies of the pop-cosmos ones, so the forecast\n"
              "      code imports, but any dust-model mismatch test (mismatch_nag=) is\n"
              "      meaningless until a real dust_choice=2 run exists. Pass --genuine-nag\n"
              "      once redshifts/nz*_nag.npy hold a true Nagaraj-dust ensemble.")

    # likelihood.py can also ask for the no-interloper coefficient means
    for d in "ugr":
        shutil.copyfile(os.path.join(path, "4pca_data", f"npca_means_{d}.npy"),
                        os.path.join(path, "4pca_data", f"npca_noint_means_{d}.npy"))

    for d in "ugr":
        C = np.load(os.path.join(path, "4pca_data", f"npca_cov_{d}.npy"))
        print(f"   {d}-dropouts: covariance {C.shape}, condition number {np.linalg.cond(C):.3g}")


if __name__ == "__main__":
    main()
