# This module contains implementations of galaxy bias
import jax.numpy as np
from jax.tree_util import register_pytree_node_class

import jax_cosmo.background as bkgrd
from jax_cosmo.jax_utils import container
from jax_cosmo.utils import a2z
from jax_cosmo.utils import z2a


@register_pytree_node_class
class custom_bias(container):
    """
    Class representing a linear bias

    Parameters:
    -----------
    b: redshift independent bias value
    """

    def __call__(self, cosmo, z):
        b = self.params[0]
        return np.where(z < 1.5, np.ones_like(z) * 1.0, np.ones_like(z) * b)
    
@register_pytree_node_class
class increasing_bias(container):
    """
    Class representing a linear bias

    Parameters:
    -----------
    b: redshift independent bias value
    """

    def __call__(self, cosmo, z):
        b = self.params[0]
        return b*(1+z) #np.where(z < 1.5, np.ones_like(z) * 1.0, np.ones_like(z) * b)


@register_pytree_node_class
class constant_linear_bias(container):
    """
    Class representing a linear bias

    Parameters:
    -----------
    b: redshift independent bias value
    """

    def __call__(self, cosmo, z):
        b = self.params[0]
        return b * np.ones_like(z)


@register_pytree_node_class
class inverse_growth_linear_bias(container):
    """
    TODO: what's a better name for this?
    Class representing an inverse bias in 1/growth(a)

    Parameters:
    -----------
    cosmo: cosmology
    b: redshift independent bias value at z=0
    """

    def __call__(self, cosmo, z):
        b = self.params[0]
        return b / bkgrd.growth_factor(cosmo, z2a(z))


@register_pytree_node_class
class des_y1_ia_bias(container):
    """
    https://arxiv.org/pdf/1708.01538.pdf Sec. VII.B

    Parameters:
    -----------
    cosmo: cosmology
    A: amplitude
    eta: redshift dependent slope
    z0: pivot redshift
    """

    def __call__(self, cosmo, z):
        A, eta, z0 = self.params
        return A * ((1.0 + z) / (1.0 + z0)) ** eta


@register_pytree_node_class
class w_w_quadratic_bias(container):
    """
    
    http://arxiv.org/abs/1904.13378 2.5.1

    Parameters:
    -----------
    cosmo: cosmology
    b0: redshift independent bias value at z=4, m=25, fiducial value 4.8
    
    """
    
    def __call__(self, cosmo, z):
        b0 = self.params[0]
        # clamp above z=8: the n(z) grid ends at ~7, but the Limber integral
        # runs to z=2000 (CMB lensing), where (1+z)^2 would blow up
        z = np.minimum(z, 8.0)
        return b0 * (0.023*(1+z) + 0.035*(1+z)**2)