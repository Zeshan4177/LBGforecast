import jax
# jax >= 0.4.25 exposes the config object on the top-level module
try:
    from jax.config import config
except ModuleNotFoundError:
    config = jax.config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax_cosmo as jc
from jax_cosmo import Cosmology
from jax import jacfwd
from functools import partial


from lbg_forecast.angular_power import cl_theory_CMB
from lbg_forecast.angular_power import cl_data_CMB
from lbg_forecast.angular_power import cl_data_CMB_nagaraj
from lbg_forecast.angular_power import compare_cls
from lbg_forecast.angular_power import define_cosmo
from lbg_forecast.angular_power import pk
from lbg_forecast.angular_power import pk_lin

from lbg_forecast.modified_likelihood import gaussian_log_likelihood
from lbg_forecast.modified_likelihood import marginalised_log_likelihood
import lbg_forecast.utils as utils


class Likelihood:
    def __init__(self, path, n_override=None, mismatch_nag=None, override_seed=None, no_noise=False):
        """
        P - N(z) covariance of PCA parameters
        C - Data covariance (includes cosmic variance + cut sky)
        ----------------------------------------------------------
        _mean_vec_u, _mean_vec_g, _mean_vec_r - mean of pca coefficients
        _cov_u, _cov_g, _cov_r - covariance matrix of pca coefficients

        """
        print("Initialising likelihood")

        self._mean_vec_u = jnp.load(path+"/4pca_data/npca_means_u.npy")
        self._mean_vec_g = jnp.load(path+"/4pca_data/npca_means_g.npy")
        self._mean_vec_r = jnp.load(path+"/4pca_data/npca_means_r.npy")

        self._cov_u = jnp.load(path+"/4pca_data/npca_cov_u.npy")
        self._cov_g = jnp.load(path+"/4pca_data/npca_cov_g.npy")
        self._cov_r = jnp.load(path+"/4pca_data/npca_cov_r.npy")

        if(mismatch_nag is not None):
            '''Data has nag dust, theory has popcosmos dust'''
            self._mean_vec_u_pop = jnp.load(path+"/4pca_data/npca_means_u.npy")
            self._mean_vec_g_pop = jnp.load(path+"/4pca_data/npca_means_g.npy")
            self._mean_vec_r_pop = jnp.load(path+"/4pca_data/npca_means_r.npy")

            self._cov_u_pop = jnp.load(path+"/4pca_data/npca_cov_u.npy")
            self._cov_g_pop = jnp.load(path+"/4pca_data/npca_cov_g.npy")
            self._cov_r_pop = jnp.load(path+"/4pca_data/npca_cov_r.npy")

            self.nz_params_mean_pop = jnp.hstack(
                (self._mean_vec_u_pop, self._mean_vec_g_pop, self._mean_vec_r_pop)
            )

            self._mean_vec_u = jnp.load(path+"/4pca_data/npca_means_u_nag.npy")
            self._mean_vec_g = jnp.load(path+"/4pca_data/npca_means_g_nag.npy")
            self._mean_vec_r = jnp.load(path+"/4pca_data/npca_means_r_nag.npy")

            self._cov_u = jnp.load(path+"/4pca_data/npca_cov_u_nag.npy")
            self._cov_g = jnp.load(path+"/4pca_data/npca_cov_g_nag.npy")
            self._cov_r = jnp.load(path+"/4pca_data/npca_cov_r_nag.npy")

        self._npca = len(self._mean_vec_u)

        zero_block = jnp.zeros((self._npca, self._npca))
        self.P = jnp.block(
            [
                [self._cov_u, zero_block, zero_block],
                [zero_block, self._cov_g, zero_block],
                [zero_block, zero_block, self._cov_r],
            ]
        )
        self._inv_P = jnp.linalg.inv(self.P)

        self._ell = jnp.arange(200, 1000, 1)
        self._fsky = 0.35
        seed = 100
        if(override_seed is not None):
            seed = override_seed

        self._b_lbg_u = 3.0#/(1+3)
        self._b_lbg_g = 4.0#/(1+4)
        self._b_lbg_r = 5.0#/(1+5)

        self.b_lbg = 3.585/(1+3.585)

        self._bias_params = jnp.array([self._b_lbg_u,
                                       self._b_lbg_g,
                                       self._b_lbg_r,
        ])

        self.nz_params_mean = jnp.hstack(
            (self._mean_vec_u, self._mean_vec_g, self._mean_vec_r)
        )

        self.nden_u = 8000/utils.DEG2_TO_ARCMIN2
        self.nden_g = 14000/utils.DEG2_TO_ARCMIN2
        self.nden_r = 1100/utils.DEG2_TO_ARCMIN2

        if n_override is not None:
            if n_override == 'max':
                self.nden_u = 10000/utils.DEG2_TO_ARCMIN2
                self.nden_g = 18000/utils.DEG2_TO_ARCMIN2
                self.nden_r = 1900/utils.DEG2_TO_ARCMIN2

            if n_override == 'min':
                self.nden_u = 6000/utils.DEG2_TO_ARCMIN2
                self.nden_g = 10000/utils.DEG2_TO_ARCMIN2
                self.nden_r = 300/utils.DEG2_TO_ARCMIN2

        self.ndens = jnp.array([self.nden_u, self.nden_g, self.nden_r])

        self._cosmo_fid = define_cosmo()

        _o_m = self._cosmo_fid.Omega_c + self._cosmo_fid.Omega_b
        _s8 = self._cosmo_fid.sigma8*jnp.sqrt(_o_m/0.3)

        self._derived_params = jnp.array([_o_m, _s8])

        # Generate mock data
        if(mismatch_nag is not None):
            mean_cl, covmat = cl_data_CMB_nagaraj(
                self._cosmo_fid,
                self.nz_params_mean,
                self._bias_params,
                self._ell,
                self._fsky,
                self.ndens,
                seed
            )
        else:
            mean_cl, covmat = cl_data_CMB(
            self._cosmo_fid,
            self.nz_params_mean,
            self._bias_params,
            self._ell,
            self._fsky,
            self.ndens,
            seed
            )

        self.cl_mean = mean_cl
        if(no_noise==True):
            self.cl_mean = cl_theory_CMB(self._cosmo_fid, self.nz_params_mean, self._bias_params, self._ell, self.ndens, red=1.0)

        # data covariance
        self.C = covmat
        self._inv_C = jnp.linalg.inv(self.C)
        self.det_C = jnp.linalg.det(self.C)

        # jacobian #need to change if you want uncertanties with nagaraj
        self._jacobian = jax.jit(jacfwd(cl_theory_CMB, argnums=1))
        self.T = self._jacobian(self._cosmo_fid, self.nz_params_mean,
                                 self._bias_params, self._ell, self.ndens, 1.0)
        
        self.Cm = self.C + self.T @ self.P @ self.T.T

        print("Initialisation Complete")

    def mu_vec_ww(self, params, red=1.0):
        """Reduced theory vector for fisher forecast"""

        ####Stuff for W&W (convert z=2.6 sigma8 to z=0.0)
        norm_diff = pk_lin(self._cosmo_fid, 1/8, 0.0)/pk_lin(self._cosmo_fid, 1/8, 2.6)
        ####
        cosmo_obj = jc.Planck15(sigma8=params[0]*jnp.sqrt(norm_diff))
        bias_params = self._bias_params
        bias_params = bias_params.at[0].set(params[1])
        bias_params = bias_params.at[3].set(params[1])
        nz_params = self.nz_params_mean
    
        return cl_theory_CMB(cosmo_obj, nz_params, bias_params, self._ell, self.ndens, red=red)
    
    def mu_vec_sig(self, params):
        """Reduced theory vector for fisher forecast"""

        ####Stuff for W&W (convert z=2.6 sigma8 to z=0.0)
        norm_diff = pk_lin(self._cosmo_fid, 1/8, 0.0)/pk_lin(self._cosmo_fid, 1/8, 2.6)
        ####
        cosmo_obj = jc.Planck15(sigma8=params[0]*jnp.sqrt(norm_diff))
        bias_params = self._bias_params
        bias_params = bias_params.at[0].set(params[1])
        bias_params = bias_params.at[3].set(params[1])
        nz_params = self.nz_params_mean
    
        return cl_theory_CMB(cosmo_obj, nz_params, bias_params, self._ell, self.ndens, red=1.0)
    
    def mu_vec(self, params, red=1.0):
        """Reduced theory vector for fisher forecast"""

        cosmo_obj = jc.Planck15(sigma8=params[0],
                                Omega_c=params[1],
                                Omega_b=params[2],
                                h=params[3],
                                n_s=params[4])

        bias_params = self._bias_params
        bias_params = bias_params.at[0].set(params[5])
        bias_params = bias_params.at[1].set(params[6])
        bias_params = bias_params.at[2].set(params[7])
        nz_params = self.nz_params_mean
    
        return cl_theory_CMB(cosmo_obj, nz_params, bias_params, self._ell, self.ndens, red=red)
    
    def mu_vec_nag(self, params):
        """Reduced theory vector for fisher forecast
        USE ONLY WITH mismatch_nag=True"""

        cosmo_obj = jc.Planck15(sigma8=params[0],
                                Omega_c=params[1],
                                Omega_b=params[2],
                                h=params[3],
                                n_s=params[4])

        bias_params = self._bias_params
        bias_params = bias_params.at[0].set(params[5])
        bias_params = bias_params.at[1].set(params[6])
        bias_params = bias_params.at[2].set(params[7])
        nz_params = self.nz_params_mean_pop
    
        return cl_theory_CMB(cosmo_obj, nz_params, bias_params, self._ell, self.ndens, red=1.0)
    
    def mu_vec_deriv(self, params, red=1.0):
        """Reduced theory vector for fisher forecast"""

        o_m = params[0]
        s8 = params[1]

        cosmo_obj = jc.Planck15(sigma8=s8/jnp.sqrt(o_m/0.3),
                                Omega_c=o_m-params[2],
                                Omega_b=params[2],
                                h=params[3],
                                n_s=params[4])

        bias_params = self._bias_params
        bias_params = bias_params.at[0].set(params[5])
        bias_params = bias_params.at[1].set(params[6])
        bias_params = bias_params.at[2].set(params[7])
        nz_params = self.nz_params_mean
    
        return cl_theory_CMB(cosmo_obj, nz_params, bias_params, self._ell, self.ndens, red=red)

    def logL(self, params):
        """marginalised likelihood"""

        cosmo_obj = jc.Planck15(sigma8=params[0],
                        Omega_c=params[1],
                        Omega_b=params[2],
                        h=params[3],
                        n_s=params[4])

        bias_params = self._bias_params
        bias_params = bias_params.at[0].set(params[5])
        bias_params = bias_params.at[1].set(params[6])
        bias_params = bias_params.at[2].set(params[7])

        nz_params = self.nz_params_mean

        T = self.T
        C = self.C
        P = self.P

        t = cl_theory_CMB(cosmo_obj, nz_params, bias_params, self._ell)
        c = self.cl_mean

        return marginalised_log_likelihood(c, t, C, P, T)

    
    def fisher(self, params, red=1.0):

        inv_cov = jnp.linalg.inv(self.C)
        mu_vec_fixed = partial(self.mu_vec, red=red)
        jac_at_mean = jax.jit(jax.jacfwd(mu_vec_fixed, argnums=0))
        dmudp = jac_at_mean(params)

        F = dmudp.T@inv_cov@dmudp

        return F
    
    def fisher_marg(self, params, red=1.0):

        inv_cov = jnp.linalg.inv(self.Cm)
        mu_vec_fixed = partial(self.mu_vec, red=red)
        jac_at_mean = jax.jit(jax.jacfwd(mu_vec_fixed, argnums=0))
        dmudp = jac_at_mean(params)

        F = dmudp.T@inv_cov@dmudp

        return F
    
    def fisher_deriv(self, params, red=1.0):

        inv_cov = jnp.linalg.inv(self.C)
        mu_vec_fixed = partial(self.mu_vec_deriv, red=red)
        jac_at_mean = jax.jit(jax.jacfwd(mu_vec_fixed, argnums=0))
        dmudp = jac_at_mean(params)

        F = dmudp.T@inv_cov@dmudp

        return F
    
    def fisher_marg_deriv(self, params, red=1.0):

        inv_cov = jnp.linalg.inv(self.Cm)
        mu_vec_fixed = partial(self.mu_vec_deriv, red=red)
        jac_at_mean = jax.jit(jax.jacfwd(mu_vec_fixed, argnums=0))
        dmudp = jac_at_mean(params)

        F = dmudp.T@inv_cov@dmudp

        return F
    
    def fisher_ww(self, params, red=1.0):

        inv_cov = jnp.linalg.inv(self.C)
        mu_vec_fixed = partial(self.mu_vec_ww, red=red)
        jac_at_mean = jax.jit(jax.jacfwd(mu_vec_fixed, argnums=0))
        dmudp = jac_at_mean(params)

        F = dmudp.T@inv_cov@dmudp

        return F
    
    def fisher_sig(self, params):

        inv_cov = jnp.linalg.inv(self.C)
        jac_at_mean = jax.jit(jax.jacfwd(self.mu_vec_ww, argnums=0))
        dmudp = jac_at_mean(params)

        F = dmudp.T@inv_cov@dmudp

        return F

    def plot_data_cls(self):
        """Plot mock data used for inference that was initalised with class"""

        # data initilaised in lhood
        data_cl = self.cl_mean

        # theory vector evaluated at mean redshift distribution
        nz_params = self.nz_params_mean
        cosmo = self._cosmo_fid
        bias_params = self._bias_params
        theory_cl = cl_theory_CMB(cosmo, nz_params, bias_params, self._ell, self.ndens)

        # plot together
        compare_cls(data_cl, theory_cl, self._ell, figure_size=(15, 10), fontsize=18, ncls=4)
