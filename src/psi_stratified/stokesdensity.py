#!/usr/bin/python
#
# Copyright 2020 Colin McNally, Sijme-Jan Paardekooper, Francesco Lovascio
#    colin@colinmcnally.ca, s.j.paardekooper@qmul.ac.uk, f.lovascio@qmul.ac.uk
#
#    This file is part of psitools.
#
#    psitools is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    psitools is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with psitools.  If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
import scipy.integrate as integrate


class StokesDensity():
    def __init__(self, stokes_range, param_dict=None):

        stokes_range = np.atleast_1d(stokes_range)

        if len(stokes_range) == 1:
            #print('Creating monodisperse StokesDensity')
            self.stokes_min = None
            self.stokes_max = stokes_range[0]
            self.f = lambda x: self.stokes_max
        elif len(stokes_range) == 2:
            #print('Creating polydisperse StokesDensity')
            self.stokes_min = stokes_range[0]
            self.stokes_max = stokes_range[1]

            # Default: MRN power law. TODO: other distributions via dict
            self.sigma0 = lambda s: np.power(s, -0.5)

            # Normalization factor
            norm = integrate.quad(self.sigma0,
                                  self.stokes_min,
                                  self.stokes_max)[0]
            self.f = \
              lambda x: self.stokes_max*self.sigma0(self.stokes_max*x)/norm
        else:
            raise ValueError('stokes_range needs to have either 1 or 2 elements')

        # Empty list of poles
        #self.poles = []

    #def __call__(self, x):
    #    """Return stokes_max*sigma(stokes_max*x)/norm"""
    #    return self.f(x)

    def sigma(self, x):
        return self.f(x/self.stokes_max)/self.stokes_max

    def integrate(self, func):
        """Integrate sigma*func over all sizes"""
        if self.stokes_min is not None:
            g = lambda x: self.f(x)*func(self.stokes_max*x)
            return integrate.fixed_quad(g,
                                        self.stokes_min/self.stokes_max,
                                        1,
                                        n=50)[0]
        else:
            return func(self.stokes_max)
