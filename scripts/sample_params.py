"""Draw SPS parameters for N prior realisations (runs in the base env)."""
import sys, time
import numpy as np
import lbg_forecast.population_model as pop
import lbg_forecast.priors_gp_massfunc as gpmf
import lbg_forecast.priors_gp_dust as gpdp
import lbg_forecast.priors_gp_csfrd as gpsf

repo, nreal, ngals, dust_choice, out_file = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
mean = int(sys.argv[6]) if len(sys.argv) > 6 else 0

mf = gpmf.MassFunctionPrior(path=repo, mean=bool(mean))
dp = gpdp.DustPrior(path=repo, mean=bool(mean))
sf = gpsf.CSFRDPrior(path=repo)

out = np.zeros((nreal, ngals, 17))
for n in range(nreal):
    t0 = time.time()
    out[n] = pop.generate_sps_parameters(ngals, mf, dp, sf, mean=mean, dust_choice=dust_choice,
                                         uniform_redshift_mass=False)
    print("  realisation %d/%d (%.0f s)" % (n + 1, nreal, time.time() - t0), flush=True)
np.save(out_file, out)
print("wrote", out_file, out.shape)
