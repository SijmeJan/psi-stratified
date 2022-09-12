# -*- coding: utf-8 -*-
"""Module containing StokesDensity class.
"""

import numpy as np
from scipy import integrate

class StokesDensity():
    """Class describing distribution of dust over Stokes number.

    Attributes:
        stokes_min: minimum Stokes number. In case of monodisperse dust, it
            is set to None.
        stokes_max: maximum Stokes number.
        param: dictionary containing additional parameters for Stokes density.
        sigma_norm: Stokes density function (normalized).
    """

    def __init__(self, stokes_range, param_dict=None):
        """Create StokesDensity instance.

        Args:
            stokes_range: [minimum, maximum] Stokes number in distribution
            param_dict: dictionary containing additional information about
                the distribution

        """

        stokes_range = np.atleast_1d(stokes_range)

        self.param = param_dict

        if len(stokes_range) == 1:
            self.stokes_min = None
            self.stokes_max = stokes_range[0]
            #self.f = lambda x: self.stokes_max
        elif len(stokes_range) == 2:
            self.stokes_min = stokes_range[0]
            self.stokes_max = stokes_range[1]

            # Default: MRN power law. TODO: other distributions via dict
            def sigma0(stk):
                return np.power(stk, -0.5)

            # Normalization factor
            norm = integrate.quad(sigma0,
                                  self.stokes_min,
                                  self.stokes_max)[0]

            self.sigma_norm = lambda s: sigma0(s)/norm
            #self.f = \
            #  lambda x: self.stokes_max*self.sigma0(self.stokes_max*x)/self.norm
        else:
            raise ValueError('stokes_range needs to have either 1 or 2 elements')

    def sigma(self, stokes_number):
        """Calculate normalized Stokes density.

        Args:
            stokes_number: Stokes number at which Stokes density is required

        Returns:
            Normalized Stokes density at required Stokes number. In the
            monodisperse limit, returns 1.

        """

        if self.stokes_min is None:
            # Monodisperse limit
            return 1

        return self.sigma_norm(stokes_number)
        #return self.f(x/self.stokes_max)/self.stokes_max

    def integrate(self, func):
        """Integrate sigma*func over all Stokes numbers.

        Args:
            func: function to integrate.

        Returns:
            Integral of sigma*func over all Stokes numbers.

        """

        if self.stokes_min is None:
            # Monodisperse limit
            return func(self.stokes_max)

        #g = lambda x: self.f(x)*func(self.stokes_max*x)
        return integrate.fixed_quad(lambda x: self.stokes_max*\
            self.sigma_norm(self.stokes_max*x)*func(self.stokes_max*x),
                                    self.stokes_min/self.stokes_max,
                                    1,
                                    n=50)[0]
