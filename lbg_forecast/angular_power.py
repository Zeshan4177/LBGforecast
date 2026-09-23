import jax
# jax >= 0.4.25 exposes the config object on the top-level module
try:
    from jax.config import config
except ModuleNotFoundError:
    config = jax.config

config.update("jax_enable_x64", True)
from jax import jit
import jax.numpy as jnp

import jax_cosmo as jc
from jax_cosmo import Cosmology

from jax_cosmo.redshift import delta_nz

from jax_cosmo.power import nonlinear_matter_power
from jax_cosmo.power import linear_matter_power

from jax_cosmo.utils import z2a

from lbg_forecast import modified_probes
from jax_cosmo import probes
from lbg_forecast import modified_angular_cl
from lbg_forecast.modified_angular_cl import noise_cl
from lbg_forecast.modified_angular_cl import gaussian_cl_covariance_and_mean

from lbg_forecast.modified_bias import custom_bias
from lbg_forecast.modified_bias import constant_linear_bias
from lbg_forecast.modified_bias import increasing_bias
from lbg_forecast.modified_bias import w_w_quadratic_bias
from lbg_forecast.modified_bias import growth_bias_I

from lbg_forecast.modified_redshift import u_dropout
from lbg_forecast.modified_redshift import g_dropout
from lbg_forecast.modified_redshift import r_dropout

from lbg_forecast.modified_redshift import u_dropout_nagaraj
from lbg_forecast.modified_redshift import g_dropout_nagaraj
from lbg_forecast.modified_redshift import r_dropout_nagaraj

from lbg_forecast.modified_redshift import histogram_nz
from lbg_forecast.modified_angular_cl import angular_cl as new_cl


from functools import partial

import matplotlib.pyplot as plt

# Number of PCA coefficients used per dropout sample. Read from the compiled
# n(z) artifacts so the code follows whatever ensemble is on disk (the published
# analysis used 50, which needs an ensemble of many more than 50 realisations).
def _npca_from_artifacts(default=50):
    import os
    import numpy as _np
    for _p in ("./4pca_data/npca_means_u.npy",
               os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "4pca_data", "npca_means_u.npy")):
        try:
            return len(_np.load(_p))
        except Exception:
            continue
    return default


NPCA = _npca_from_artifacts()

def define_cosmo():
    """
    Define a cosmology in jax-cosmo (Planck 2015 results)

    """
    return jc.Planck15()


def z_space():
    """
    Redshift space grid for redshift distributions

    """
    return jnp.arange(0, 7, 0.01)


def limber_weight(cosmo, z, kernel_i, kernel_j):
    """
    Limber integrand of C_ell^{ij} in z, without P(k, z=0):
    w_ij(z) = K_i(z) K_j(z) D^2(z) / (H(z) chi^2(z))
    This is the angular_cl integrand, with dchi/dz = c/H and the constant
    factors dropped, since only ratios of integrals of w are used.
    --------------------------------------------------------------------
    Parameters:
    cosmo - JAX-COSMO cosmology object containing cosmological parameters
    z - redshifts
    kernel_i, kernel_j - probe kernels evaluated at z (as returned by probe.kernel)
    ----------------------------------------------------------------------
    Returns:
    w_ij(z)

    """
    a = z2a(z)
    chi = jc.background.radial_comoving_distance(cosmo, a)
    D = jc.background.growth_factor(cosmo, a)
    return kernel_i * kernel_j * D**2 / (jc.background.H(cosmo, a) * jnp.clip(chi**2, 1.0))


def z_eff(cosmo, nz_params, ndens, red=1.0, z_min=1.5):
    """
    Effective redshift of each dropout sample, excluding interlopers,
    weighted by the full (f_NL = 0) Limber kernel of the auto spectrum:
    z_eff = int_{z_min} z w(z) dz / int_{z_min} w(z) dz,
    w(z) = n^2(z) b^2(z) D^2(z) H(z) / chi^2(z)
    Above z_min the LBG bias is b_0 (1+z)/(1+z_eff), so only its (1+z) shape
    enters w and the pivot z_eff cancels (no circularity).
    --------------------------------------------------------------------
    Parameters:
    cosmo - JAX-COSMO cosmology object containing cosmological parameters
    nz_params - PCA coefficients for u, g, r dropout redshift distributions
    ndens - number densities of u, g, r dropouts
    red - interloper reduction factor
    z_min - lower integration limit (interloper cut)
    ----------------------------------------------------------------------
    Returns:
    jnp.array([z_eff_u, z_eff_g, z_eff_r])

    """
    n = NPCA

    red = jnp.array([red])
    nz_u = u_dropout(nz_params[:n], gals_per_arcmin2=ndens[0], red=red)
    nz_g = g_dropout(nz_params[n : 2 * n], gals_per_arcmin2=ndens[1], red=red)
    nz_r = r_dropout(nz_params[2 * n : 3 * n], gals_per_arcmin2=ndens[2], red=red)

    z = z_space()
    z = z[z >= z_min]

    z_effs = []
    for nz in [nz_u, nz_g, nz_r]:
        # density kernel n(z) b(z) H(z), with the (1+z) shape of the LBG bias
        kernel = nz(z) * (1 + z) * jc.background.H(cosmo, z2a(z))
        w = limber_weight(cosmo, z, kernel, kernel)
        z_effs.append(jnp.trapezoid(z * w, z) / jnp.trapezoid(w, z))

    return jnp.array(z_effs)


@jit
def cl_theory_CMB(cosmo, nz_params, bias_params, ell, ndens, red):
    """
    Calculates theory vector for Likelihood. Computes angular cls
    and cross correlations of u, g, r dropouts with two component bias.
    --------------------------------------------------------------------
    Parameters:
    cosmo - JAX-COSMO cosmology object containing cosmological parameters
    nz_params -
    b_int - Interloper bias (linear)
    b_lbg - LBG bias (linear)
    ell - Spherical harmonic scale list. Gives range of ells to plot cls over
    ----------------------------------------------------------------------
    Returns:
    Concatenated angular power spectra of length 6*len(ell) giving auto+cross
    spectra, with poisson noise

    """
    n = NPCA

    surface_of_last_scattering = delta_nz(1100., gals_per_arcmin2 = 1e20, zmax=2000.) 
    red = jnp.array([red])
    nz_u = u_dropout(nz_params[:n], gals_per_arcmin2=ndens[0], red=red)#1
    nz_g = g_dropout(nz_params[n : 2 * n], gals_per_arcmin2=ndens[1], red=red)#1
    nz_r = r_dropout(nz_params[2 * n : 3 * n], gals_per_arcmin2=ndens[2], red=red)#0.1

    redshift_distributions = [nz_u, nz_g, nz_r]

   #bias = [
   #     increasing_bias(bias_params[0]),
   #     increasing_bias(bias_params[1]),
   #     increasing_bias(bias_params[2]),
   # ]

    #bias = [
    #    constant_linear_bias(bias_params[0]),
    #    constant_linear_bias(bias_params[1]),
    #    constant_linear_bias(bias_params[2]),
    #]

    # bias_params = [b_0_u, b_0_g, b_0_r, b_I, z_eff_u, z_eff_g, z_eff_r, f_NL],
    # one interloper amplitude b_I shared by all three samples, interloper bias = b_I/D(z)
    bias = [
        growth_bias_I(bias_params[0], bias_params[3], bias_params[4]),
        growth_bias_I(bias_params[1], bias_params[3], bias_params[5]),
        growth_bias_I(bias_params[2], bias_params[3], bias_params[6]),
    ]

    cosmo_probes = [modified_probes.NumberCounts(redshift_distributions, bias, bias_params[7]),
                    modified_probes.WeakLensing([surface_of_last_scattering])]

    signal = modified_angular_cl.angular_cl(cosmo, ell, cosmo_probes)
    noise = noise_cl(ell, cosmo_probes)
    total_cl = signal + noise

    return jnp.hstack(total_cl)

@jit
def cl_data_CMB(cosmo, nz_params, bias_params, ell, f_sky, ndens, seed, red=1.0):
    """
    Genrates Mock LBG lustering angular power spectra data. Gives
    u, g, r-dropout clustering plus cross correlations. Gaussian noise
    is added to simulate cosmic variance, which is also scaled by
    sky fraction.
    --------------------------------------------------------------------
    Parameters:
    cosmo - JAX-COSMO cosmology object containing cosmological parameters
    nz_params -
    b_int - Interloper bias (linear)
    b_lbg - LBG bias (linear)
    ell - Spherical harmonic scale list. Gives range of ells to plot cls over
    ----------------------------------------------------------------------
    Returns:
    Concatenated angular power spectra of length 6*len(ell) giving auto+cross
    spectra, with poisson noise, cosmic variance plus contribution from cut sky

    """
    n = NPCA

    surface_of_last_scattering = delta_nz(1100., gals_per_arcmin2 = 1e20, zmax=2000.)

    nz_u = u_dropout(nz_params[:n], gals_per_arcmin2=ndens[0], red=red)
    nz_g = g_dropout(nz_params[n : 2 * n], gals_per_arcmin2=ndens[1], red=red)
    nz_r = r_dropout(nz_params[2 * n : 3 * n], gals_per_arcmin2=ndens[2], red=red)

    redshift_distributions = [nz_u, nz_g, nz_r]

   #bias = [
   #     increasing_bias(bias_params[0]),
   #     increasing_bias(bias_params[1]),
   #     increasing_bias(bias_params[2]),
   # ]

    #bias = [
    #    constant_linear_bias(bias_params[0]),
    #    constant_linear_bias(bias_params[1]),
    #    constant_linear_bias(bias_params[2]),
    #]

    # bias_params = [b_0_u, b_0_g, b_0_r, b_I, z_eff_u, z_eff_g, z_eff_r, f_NL],
    # one interloper amplitude b_I shared by all three samples, interloper bias = b_I/D(z)
    bias = [
        growth_bias_I(bias_params[0], bias_params[3], bias_params[4]),
        growth_bias_I(bias_params[1], bias_params[3], bias_params[5]),
        growth_bias_I(bias_params[2], bias_params[3], bias_params[6]),
    ]

    cosmo_probes = [modified_probes.NumberCounts(redshift_distributions, bias, bias_params[7]),
                    modified_probes.WeakLensing([surface_of_last_scattering])]

    signal, cov = gaussian_cl_covariance_and_mean(
        cosmo, ell, cosmo_probes, f_sky=f_sky, sparse=False
    )

    noise = jnp.hstack(noise_cl(ell, cosmo_probes))

    total_cl = signal + noise
    
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    #total_cl = total_cl*jax.random.chisquare(key=subkey, df=(2.*f_sky*jnp.repeat(ell, repeats=10)+1))/(2.*jnp.repeat(ell, repeats=10)*f_sky+1)
    total_cl = jax.random.multivariate_normal(key=subkey, mean=total_cl, cov=cov)
    return total_cl, cov

#@jit
def cl_data_CMB_nagaraj(cosmo, nz_params, bias_params, ell, f_sky, ndens, seed, red=1.0):
    """
    Genrates Mock LBG lustering angular power spectra data. Gives
    u, g, r-dropout clustering plus cross correlations. Gaussian noise
    is added to simulate cosmic variance, which is also scaled by
    sky fraction.
    --------------------------------------------------------------------
    Parameters:
    cosmo - JAX-COSMO cosmology object containing cosmological parameters
    nz_params -
    b_int - Interloper bias (linear)
    b_lbg - LBG bias (linear)
    ell - Spherical harmonic scale list. Gives range of ells to plot cls over
    ----------------------------------------------------------------------
    Returns:
    Concatenated angular power spectra of length 6*len(ell) giving auto+cross
    spectra, with poisson noise, cosmic variance plus contribution from cut sky

    """
    n = NPCA

    surface_of_last_scattering = delta_nz(1100., gals_per_arcmin2 = 1e20) 

    nz_u = u_dropout_nagaraj(nz_params[:n], gals_per_arcmin2=ndens[0], red=red)
    nz_g = g_dropout_nagaraj(nz_params[n : 2 * n], gals_per_arcmin2=ndens[1], red=red)
    nz_r = r_dropout_nagaraj(nz_params[2 * n : 3 * n], gals_per_arcmin2=ndens[2], red=red)

    redshift_distributions = [nz_u, nz_g, nz_r]

   #bias = [
   #     increasing_bias(bias_params[0]),
   #     increasing_bias(bias_params[1]),
   #     increasing_bias(bias_params[2]),
   # ]

    #bias = [
    #    constant_linear_bias(bias_params[0]),
    #    constant_linear_bias(bias_params[1]),
    #    constant_linear_bias(bias_params[2]),
    #]

    # bias_params = [b_0_u, b_0_g, b_0_r, b_I, z_eff_u, z_eff_g, z_eff_r, f_NL],
    # one interloper amplitude b_I shared by all three samples, interloper bias = b_I/D(z)
    bias = [
        growth_bias_I(bias_params[0], bias_params[3], bias_params[4]),
        growth_bias_I(bias_params[1], bias_params[3], bias_params[5]),
        growth_bias_I(bias_params[2], bias_params[3], bias_params[6]),
    ]
    cosmo_probes = [modified_probes.NumberCounts(redshift_distributions, bias, bias_params[7]),
                    modified_probes.WeakLensing([surface_of_last_scattering])]

    signal, cov = gaussian_cl_covariance_and_mean(
        cosmo, ell, cosmo_probes, f_sky=f_sky, sparse=False
    )

    noise = jnp.hstack(noise_cl(ell, cosmo_probes))

    total_cl = signal + noise

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    #total_cl = total_cl*jax.random.chisquare(key=subkey, df=(2.*f_sky*jnp.repeat(ell, repeats=10)+1))/(2.*jnp.repeat(ell, repeats=10)*f_sky+1)
    total_cl = jax.random.multivariate_normal(key=subkey, mean=total_cl, cov=cov)

    return total_cl, cov

def plot_ncls(cls_theory, ell, figure_size, fontsize, ncls):
    """
    Plots auto and cross power spectra in a triangle plot.
    ----------------------------------------------------------
    Parameters:
    cls_theory - Concatenated theory vector from cl_theory()
    ell - Spherical harmonic scale list. Gives range of ells to plot cls over
    figure_size, fontsize - plotting

    """
    tot_plots = ncls*ncls - sum(jnp.arange(0, ncls, 1))

    fig, axes = plt.subplots(ncls, ncls, figsize=figure_size)
    cl_list = jnp.split(cls_theory, tot_plots)

    i = 0
    j = 0
    k = 0
    while j < ncls:
        i = 0
        while i < ncls:
            ax = axes[i][j]
            if i >= j:
                ax.plot(ell, cl_list[k])
                # ax.set_xscale("log")
                ax.set_yscale("log")
                k += 1
            else:
                ax.set_visible(False)

            # plotting labels
            if i == 2 and j == 0:
                ax.set_ylabel("$C_{\ell}$", fontsize=fontsize)
            if i == 2 and j == 1:
                ax.set_xlabel("$\ell$", fontsize=fontsize)

            i += 1

        j += 1


def plot_tracer_pk(cosmo, nz_params, bias_params, k, ndens, figure_size, fontsize,
                   red=1.0, f_NL=None, z_min=1.5):
    """
    Plots the effective tracer power spectra <b_X b_Y> P_m(k, z_eff) for every tracer
    pair, in the same order and triangle layout as the angular cls.
    Every pair XY is weighted by its own Limber kernel (see limber_weight()),
    w_XY(z) = K_X(z) K_Y(z) D^2(z) / (H(z) chi^2(z)), with K the probe kernels for
    unit galaxy bias, so the cross spectra use the kernel overlap of the two tracers.
    The galaxy kernels are cut below z_min to exclude interlopers, as in z_eff():
    z_eff = int z w_XY b_X b_Y dz / int w_XY b_X b_Y dz (Gaussian biases, f_NL = 0)
    <b_X b_Y>(k) = int w_XY b_X(k, z) b_Y(k, z) dz / int w_XY dz
    With f_NL != 0 the bias is scale dependent, b(k, z) = b(z) + b_PNG(k, z), and the
    Gaussian (f_NL = 0) spectra are overplotted dashed. |P_XY(k)| is plotted, since
    for f_NL < 0 the bias can turn negative on large scales. CMB lensing has b = 1.
    --------------------------------------------------------------------
    Parameters:
    cosmo - JAX-COSMO cosmology object containing cosmological parameters
    nz_params - PCA coefficients for u, g, r dropout redshift distributions
    bias_params - bias parameters
                  [b_0_u, b_0_g, b_0_r, b_I, z_eff_u, z_eff_g, z_eff_r, f_NL]
    k - wavenumbers [h/Mpc] to plot over
    ndens - number densities of u, g, r dropouts
    figure_size, fontsize - plotting
    red - interloper reduction factor
    f_NL - local primordial non-Gaussianity, defaults to bias_params[7]
    z_min - lower integration limit for the galaxy kernels (interloper cut)
    ----------------------------------------------------------------------
    Returns:
    fig, axes, bb_eff, z_eff, one entry per tracer pair in the cl ordering
    (bb_eff is the Gaussian, f_NL = 0, <b_X b_Y>)

    """
    n = NPCA

    if f_NL is None:
        f_NL = bias_params[7]
    f_NL = float(f_NL)

    surface_of_last_scattering = delta_nz(1100., gals_per_arcmin2 = 1e20, zmax=2000.)
    red = jnp.array([red])
    nz_u = u_dropout(nz_params[:n], gals_per_arcmin2=ndens[0], red=red)
    nz_g = g_dropout(nz_params[n : 2 * n], gals_per_arcmin2=ndens[1], red=red)
    nz_r = r_dropout(nz_params[2 * n : 3 * n], gals_per_arcmin2=ndens[2], red=red)

    redshift_distributions = [nz_u, nz_g, nz_r]
    bias = [
        growth_bias_I(bias_params[0], bias_params[3], bias_params[4]),
        growth_bias_I(bias_params[1], bias_params[3], bias_params[5]),
        growth_bias_I(bias_params[2], bias_params[3], bias_params[6]),
    ]
    unit_bias = [constant_linear_bias(1.0)] * len(bias)

    # probe kernels with unit galaxy bias and f_NL = 0, shape [ntracers, nz]. Without
    # PNG the kernels do not depend on ell. The n(z) grid is extended to last
    # scattering, as the angular cl integral is, so the lensing kernel is complete
    z = jnp.concatenate([z_space(), jnp.geomspace(7.0, 1100.0, 400)])
    cosmo_probes = [modified_probes.NumberCounts(redshift_distributions, unit_bias, 0.0),
                    modified_probes.WeakLensing([surface_of_last_scattering])]
    kernels = jnp.vstack([p.kernel(cosmo, z, 1000.0) for p in cosmo_probes])
    # galaxy integrals start at z_min (interloper cut), as in z_eff(); CMB lensing keeps
    # its full range
    ngal = len(redshift_distributions)
    kernels = kernels.at[:ngal].set(jnp.where(z >= z_min, kernels[:ngal], 0.0))
    cl_index = modified_angular_cl._get_cl_ordering(cosmo_probes)

    # Gaussian bias b [nz] and scale dependent bias b_k [nk, nz] of each tracer
    b_z = []
    b_kz = []
    for b in bias:
        b_z.append(b(cosmo, z))
        b_png = jax.vmap(lambda kk: modified_probes.png_bias(cosmo, b_z[-1], kk, z, f_NL))(k)
        b_kz.append(b_z[-1] + b_png)

    # CMB lensing has no bias
    b_z.append(jnp.ones_like(z))
    b_kz.append(jnp.ones((len(k), len(z))))

    names = ["u", "g", "r", r"$\kappa$"]
    ntracers = len(names)

    fig, axes = plt.subplots(ntracers, ntracers, figsize=figure_size)
    for i in range(ntracers):
        for j in range(i + 1, ntracers):
            axes[i][j].set_visible(False)

    bb_eff = []
    z_effs = []
    for i, j in cl_index:
        w = limber_weight(cosmo, z, kernels[i], kernels[j])
        w_b = w * b_z[i] * b_z[j]
        z_pair = jnp.trapezoid(z * w_b, z) / jnp.trapezoid(w_b, z)
        bb = jnp.trapezoid(w_b, z) / jnp.trapezoid(w, z)
        bb_k = jnp.trapezoid(w * b_kz[i] * b_kz[j], z, axis=1) / jnp.trapezoid(w, z)
        bb_eff.append(bb)
        z_effs.append(z_pair)

        ax = axes[j][i]
        pk_pair = pk(cosmo, k, z_pair)
        ax.plot(k, jnp.abs(bb_k * pk_pair), label=r"$f_{NL}=%g$" % f_NL)
        if f_NL != 0.0:
            ax.plot(k, bb * pk_pair, ls="--", color="k", label=r"$f_{NL}=0$")
        # scales probed by the ell = 200-1000 cls at z ~ 3-5
        ax.axvspan(0.04, 0.2, color="grey", alpha=0.15)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(names[i] + r"$\times$" + names[j] + r", $z_{eff}=%.2f$" % z_pair,
                     fontsize=fontsize)

    for i in range(ntracers):
        axes[i][0].set_ylabel("$P_{XY}(k)$", fontsize=fontsize)
        axes[ntracers - 1][i].set_xlabel("$k$ [h/Mpc]", fontsize=fontsize)

    if f_NL != 0.0:
        axes[0][0].legend(fontsize=fontsize)

    fig.tight_layout()

    bb_eff = jnp.array(bb_eff)
    z_effs = jnp.array(z_effs)

    for (i, j), bb, zz in zip(cl_index, bb_eff, z_effs):
        print("%s x %s: <b b>_eff = %.3f, z_eff = %.3f" % (names[i], names[j], bb, zz))

    return fig, axes, bb_eff, z_effs


def compare_cls(cl1, cl2, ell, figure_size, fontsize, ncls):
    """
    Plots two sets of cls on one plot in order to compare
    ----------------------------------------------------------
    Parameters:
    cl1 - Concatenated cl vector 1
    cl2 - Concatenated cl vector 2
    ell - Spherical harmonic scale list. Gives range of
          ells to plot cls over

    (figure_size, fontsize - plotting)

    """
    tot_plots = ncls*ncls - sum(jnp.arange(0, ncls, 1))

    fig, axes = plt.subplots(ncls, ncls, figsize=figure_size)
    cl1_list = jnp.split(cl1, tot_plots)
    cl2_list = jnp.split(cl2, tot_plots)

    i = 0
    j = 0
    k = 0
    while j < ncls:
        i = 0
        while i < ncls:
            ax = axes[i][j]
            if i >= j:
                ax.plot(ell, cl1_list[k])
                ax.plot(ell, cl2_list[k], ls='--')
                # ax.set_xscale("log")
                ax.set_yscale("log")
                k += 1
            else:
                ax.set_visible(False)

            # plotting labels
            if i == 1 and j == 0:
                ax.set_ylabel("$C_{\ell}$", fontsize=fontsize)
            if i == 2 and j == 1:
                ax.set_xlabel("$\ell$", fontsize=fontsize)

            i += 1

        j += 1


@jit
def cl_hat(cosmo, bin_heights, bin_edges, ell):
    """function to test hist nz"""

    nz = [histogram_nz(bin_heights, bin_edges)]
    bias = constant_linear_bias(1.0)
    tracers = [modified_probes.NumberCounts(nz, bias)]

    signal = new_cl(cosmo, ell, tracers)

    return signal.flatten()

def pk(cosmo, k, z):
    return nonlinear_matter_power(cosmo, k, a=z2a(z))

def pk_lin(cosmo, k, z):
    return linear_matter_power(cosmo, k, a=z2a(z))