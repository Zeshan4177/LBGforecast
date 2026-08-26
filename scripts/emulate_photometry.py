"""Emulated LSST ugriz photometry from SPS parameters (runs in the lbgforecast_emu env).

    python emulate_photometry.py <repo> <sps_file> <out_file>
"""
import sys
import numpy as np

# the saved Photulator pickles predate TF 2.13, when the tracking wrappers moved
import tensorflow as tf
import tensorflow.python.trackable as _trackable
import tensorflow.python.trackable.data_structures as _ds
sys.modules.setdefault("tensorflow.python.training.tracking", _trackable)
sys.modules.setdefault("tensorflow.python.training.tracking.data_structures", _ds)
from speculator import Photulator

repo, sps_file, out_file = sys.argv[1], sys.argv[2], sys.argv[3]

models = [Photulator(restore=True, restore_filename=repo + f"/trained_models/model_0x0lsst_{f}")
          for f in "ugriz"]
corr_table = np.loadtxt(repo + "/corrections/wmap1_to_9.txt")

sps = np.load(sps_file)
out = np.empty((sps.shape[0], sps.shape[1], 5))
for n in range(sps.shape[0]):
    p = sps[n].astype(np.float32)
    corr = np.interp(p[:, 0], corr_table[0, :], corr_table[1, :])
    out[n] = np.column_stack([np.asarray(m.magnitudes_(p)).ravel() for m in models]) + corr[:, None]
    print("  emulated realisation %d/%d" % (n + 1, sps.shape[0]), flush=True)
np.save(out_file, out)
print("wrote", out_file, out.shape)
