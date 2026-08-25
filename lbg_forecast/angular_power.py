import jax
from jax import config

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

NPCA=50

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

    bias = [
        constant_linear_bias(bias_params[0]),
        constant_linear_bias(bias_params[1]),
        constant_linear_bias(bias_params[2]),
    ]

    cosmo_probes = [probes.NumberCounts(redshift_distributions, bias),
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

    bias = [
        constant_linear_bias(bias_params[0]),
        constant_linear_bias(bias_params[1]),
        constant_linear_bias(bias_params[2]),
    ]

    cosmo_probes = [probes.NumberCounts(redshift_distributions, bias),
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

    bias = [
        constant_linear_bias(bias_params[0]),
        constant_linear_bias(bias_params[1]),
        constant_linear_bias(bias_params[2]),
    ]
    cosmo_probes = [probes.NumberCounts(redshift_distributions, bias),
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