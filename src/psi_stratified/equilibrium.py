import numpy as np

import scipy.integrate as integrate
from scipy.special import roots_legendre
from scipy.optimize import fsolve

#from spectral_bvp.singular_bvp import SingularLinearBVP
#from spectral_bvp.basis import BoundaryCondition

from spectral_solvers.spectral_solvers import BoundaryValueProblemSolver


class EquilibriumBase():
    def __init__(self,
                 stokes_density,
                 n_dust,
                 viscous_alpha=1.0e-6,
                 dust_to_gas_ratio=1.0,
                 eta=0.05):
        """Class holding the equilibrium solution for a polydisperse dusty gas in an unstratified shearing box. All solutions are horizontally uniform (except for the background shear). Solutions in the vertical direction can be obtained analytically for the densities and vertical velocities. Horizontal velocities must be obtained numerically. Units: time in 1/Omega, length in c/Omega, gas density is unity in mid plane."""

        # Check if parameters are sensible
        if viscous_alpha <= 0.0:
            raise ValueError('Need viscous_alpha to be positive!')
        if dust_to_gas_ratio <= 0.0:
            raise ValueError('Need dust_to_gas_ratio to be positive!')

        self.stokes_density = stokes_density
        self.viscous_alpha = viscous_alpha
        self.mu = dust_to_gas_ratio
        self.eta = eta

        self.n_dust = n_dust

        # Monodisperse setup for n_dust=1
        xi = np.asarray([1.0])
        self.weights = np.asarray([1.0/self.stokes_density.stokes_max])

        # Minimum and maximum stopping time in size distribution
        taumin = self.stokes_density.stokes_max   # Monodisperse for n_dust=1
        taumax = self.stokes_density.stokes_max

        # Polydisperse setup
        if self.n_dust > 1:
            taumin = self.stokes_density.stokes_min
            xi, self.weights = roots_legendre(self.n_dust)

            # Absorb scale factor for integrals into weights
            self.weights = self.weights*0.5*np.log(taumax/taumin)

        # Gauss-Legendre nodes in terms of stopping time
        self.tau = taumin*np.power(taumax/taumin, 0.5*(xi + 1.0))

        if self.n_dust == 0:
            self.tau = []
            self.weights = []

    def beta(self, tau):
        """Vertical dust velocity is -beta*z. Valid for tau < 1/2."""
        return 0.5*(1/tau - np.sqrt(1/tau/tau - 4))

    def diff(self, tau):
        """Dust diffusion coefficient"""
        return (1 + tau + 4*tau*tau)*self.viscous_alpha/(1 + tau)/(1 + tau)

    def uz(self, z, tau=None):
        """Dust vertical velocity"""
        if tau is not None:
            # For specific tau
            return -self.beta(tau)*z
        else:
            # For all tau's
            ret = np.zeros((self.n_dust, len(z)))

            for i in range(0, self.n_dust):
                ret[i,:] = -self.beta(self.tau[i])*z

            return ret

    def duz(self, z, tau=None):
        """Dust vertical velocity, derivative wrt z"""
        if tau is not None:
            # For specific tau
            return -self.beta(tau)
        else:
            # For all tau's
            ret = np.zeros((self.n_dust, len(z)))

            for i in range(0, self.n_dust):
                ret[i,:] = -self.beta(self.tau[i])

            return ret

    def gasdens(self, z):
        """Gas density"""
        # Make sure we can handle both vector and scalar z
        z = np.asarray(z)
        scalar_input = False
        if z.ndim == 0:
            z = z[None]  # Makes z 1D
            scalar_input = True
        else:
            original_shape = np.shape(z)
            z = np.ravel(z)

        res = 0*z

        for i in range(0, len(z)):
            f = lambda tau: self.diff(tau)*(np.exp(-0.5*self.beta(tau)*z[i]*z[i]/self.diff(tau)) - 1)/tau
            res[i] = np.exp(-0.5*z[i]*z[i] + \
                            self.mu*self.stokes_density.integrate(f))

        # Return value of original shape
        if scalar_input:
            return np.squeeze(res)
        return np.reshape(res, original_shape)

    def sigma(self, z, tau=None):
        """Dust size density"""
        if tau is not None:
            # Specific tau
            return self.mu*self.stokes_density.sigma(tau)*self.gasdens(z)*np.exp(-0.5*self.beta(tau)*z*z/self.diff(tau))
        else:
            # All tau's
            ret = np.zeros((self.n_dust, len(z)))

            for i in range(0, self.n_dust):
                ret[i,:] = self.mu*self.stokes_density.sigma(self.tau[i])*self.gasdens(z)*np.exp(-0.5*self.beta(self.tau[i])*z*z/self.diff(self.tau[i]))

            return ret

    def epsilon(self, z, tau=None):
        """Dust to gas ratio"""
        if tau is not None:
            # Specific value of tau
            return self.mu*self.stokes_density.sigma(tau)*np.exp(-0.5*self.beta(tau)*z*z/self.diff(tau))
        else:
            # All tau's
            ret = np.zeros((self.n_dust, len(z)))

            #print('Hallo:', self.tau, self.stokes_density.sigma(self.tau))
            #print(self.stokes_density.integrate(lambda x: 1), self.tau*self.weights*self.stokes_density.sigma(self.tau), integrate.quad(self.stokes_density.sigma, self.stokes_density.stokes_min, self.stokes_density.stokes_max))

            for i in range(0, self.n_dust):
                ret[i, :] = self.mu*self.stokes_density.sigma(self.tau[i])*np.exp(-0.5*self.beta(self.tau[i])*z*z/self.diff(self.tau[i]))

            return ret

    def gas_surface_density(self, height=10):
        """Gas surface density"""
        res = 2*integrate.quad(self.gasdens, 0, height)[0]
        return res

    def dust_density(self, z):
        """Total dust density"""
        # Make sure we can handle both vector and scalar z
        z = np.asarray(z)
        scalar_input = False
        if z.ndim == 0:
            z = z[None]  # Makes z 1D
            scalar_input = True
        else:
            original_shape = np.shape(z)
            z = np.ravel(z)

        res = 0*z

        for i in range(0, len(z)):
            f = lambda x: self.mu*self.gasdens(z[i])*np.exp(-0.5*self.beta(x)*z[i]*z[i]/self.diff(x))
            res[i] = self.stokes_density.integrate(f)

        # Return value of original shape
        if scalar_input:
            return np.squeeze(res)
        return np.reshape(res, original_shape)

    def dust_surface_density(self, height=10):
        """Dust surface density"""
        res = 2*integrate.quad(self.dust_density, 0, height)[0]
        return res

    def metallicity(self, mu=None):
        """Metallicity = dust to gas ratio (surface densities)"""
        if mu is not None:
            self.mu = mu
        res = self.dust_surface_density()/self.gas_surface_density()
        return res

    def set_metallicity(self, Z):
        """Attempt to set metallicity"""

        # Minimize metallicity - Z
        f = lambda x: self.metallicity(mu=x) - Z

        res = fsolve(f, self.mu)[0]

        # Set dust to gas ratio to required value
        self.mu = res

    #def int_j(self, alpha):
    #    """Calculate the J_alpha integral, needed for the background velocities"""
    #    ret = 0.0
    #    for i in range(0, self.n_dust):
    #        ret += self.weights[i]*self.epsilon(0, self.tau[i])*np.power(self.tau[i], alpha + 1)/(1 + self.tau[i]*self.tau[i])

    #    return ret

    def dlogrhogdz(self, z):
        ret = -z

        mu = self.epsilon(z)
        for i in range(0, self.n_dust):
            ret -= z*self.weights[i]*self.beta(self.tau[i])*mu[i,:]
        return ret

    def dlogrhogdzdz(self, z):
        # (rhog'/rhog)'
        ret = -1 + z - z

        mu = self.epsilon(z)
        dmu = self.dlogmudz(z)

        for i in range(0, self.n_dust):
            ret -= \
              self.weights[i]*self.beta(self.tau[i])*mu[i,:]*(1 + z*dmu[i,:])
        return ret

    def dlogmudz(self, z):
        '''Logarithmic derivative dust-to-gas ratio for all species'''
        ret = np.zeros((self.n_dust, len(z)))

        for i in range(0, self.n_dust):
            ret[i, :] = -self.beta(self.tau[i])*z/self.diff(self.tau[i])

        return ret

    def d2logmudz2(self, z):
        '''Second derivative of dust-to-gas ratio/ mu for all species'''
        ret = np.zeros((self.n_dust, len(z)))

        for i in range(0, self.n_dust):
            b = self.beta(self.tau[i])
            D = self.diff(self.tau[i])
            ret[i, :] = b*(b*z*z/D - 1)/D

        return ret

    def dlogsigmadz(self, z):
        '''Logarithmic derivative size density for all species'''
        ret = self.dlogmudz(z)

        for i in range(0, self.n_dust):
            ret[i, :] += self.dlogrhogdz(z)

        return ret

class EquilibriumSolver(BoundaryValueProblemSolver):
    """Solve BVP for horizontal velocities

    Horizontal velocities are found from a BVP on an infinite interval. Spectral collocation transforms the BVP into a matrix equation M*x = S, where M is a matrix (obtained from differential operators), S is the source term, and x is the vector of unknowns. The class BoundaryValueProblemSolver contains all the necessary machinery to solve this; we only need to define M and S.
    """

    def vectorS(self, tau=[], beta=[], w=[], mu=None,
                dlogrhogdz=None, viscous_alpha=0):
        """Calculate source terms for BVP"""

        # Number of dust species
        n_dust = len(tau)

        # Number of equations: 2 gas + 2 for every dust species
        n_eq = 2 + 2*n_dust

        # Number of collocation points
        N = len(self.basis.collocation_points())

        # Full vector: N entries per equation
        ret = np.zeros((N*n_eq), dtype=np.double)

        # Only gas equations have a source
        for n in range(0, n_dust):
            ret[0*N:(0+1)*N] += 2*mu(self.z,tau[n])*w[n]*tau[n]/(1 + tau[n]**2)
            ret[1*N:(1+1)*N] -= mu(self.z,tau[n])*w[n]*tau[n]**2/(1 + tau[n]**2)

        return ret

    def matrixM(self, tau=[], beta=[], w=[], mu=None,
                dlogrhogdz=None, viscous_alpha=0):
        """Calculate matrix M for BVP M*x = S

        If viscous_alpha is zero we are neglecting gas viscosity."""

        # Number of dust species
        n_dust = len(tau)

        # Number of equations: 2 gas + 2 for every dust species
        n_eq = 2 + 2*n_dust

        # Number of collocation points
        N = len(self.basis.collocation_points())

        # Full matrix
        ret = np.zeros((N*n_eq, N*n_eq), dtype=np.double)

        #print(tau, beta, w, mu)

        # Loop through all equations
        for i in range(0, n_eq):
            # Loop through all velocities
            for j in range(0, n_eq):
                # By default, add zeros
                M = np.zeros((N, N), dtype=np.double)

                # vgx equation
                if i == 0:
                    if j == 0:
                        # vgx
                        f0 = 0*self.z
                        for n in range(0, n_dust):
                            f0 -= mu(self.z, tau[n])*w[n]
                        f1 = viscous_alpha*dlogrhogdz(self.z)
                        M = viscous_alpha*self.ddA + \
                          self.construct_m(f1, k=1) + self.construct_m(f0)
                    if j == 1:
                        # vgy
                        M = 2*self.A
                    if j > 1 and j < 2 + n_dust:
                        # u_x
                        f = w[j-2]*mu(self.z, tau[j-2])
                        M = self.construct_m(f)

                # vgy equation
                if i == 1:
                    if j == 0:
                        # vgx
                        M = -0.5*self.A
                    if j == 1:
                        # vgy
                        f0 = 0*self.z
                        for n in range(0, n_dust):
                            f0 -= mu(self.z, tau[n])*w[n]
                        f1 = viscous_alpha*dlogrhogdz(self.z)
                        M = viscous_alpha*self.ddA + \
                          self.construct_m(f1, k=1) + self.construct_m(f0)
                    if j >= 2 + n_dust:
                        # u_y
                        f = w[j-2-n_dust]*mu(self.z, tau[j-2-n_dust])
                        M = self.construct_m(f)

                # ux equations
                if i > 1 and i < 2 + n_dust:
                    if j == 0:
                        # vgx
                        M = self.A/tau[i-2]
                    if j == i:
                        # u_x
                        f1 = beta[j-2]*self.z
                        M = -self.A/tau[j-2] + self.construct_m(f1, k=1)
                    #if j >= 2 + n_dust:
                    if j == i + n_dust:
                        # u_y
                        M = 2*self.A

                # uy equations
                if i >= 2 + n_dust:
                    if j == 1:
                        # vgy
                        M = self.A/tau[i-2-n_dust]
                    if j == i - n_dust:
                        # u_x
                        M = -0.5*self.A
                    if j == i:
                        # u_y
                        f1 = beta[i-2-n_dust]*self.z
                        M = -self.A/tau[i-2-n_dust] + self.construct_m(f1, k=1)


                ret[i*N:(i+1)*N, j*N:(j+1)*N] = M

        return ret

class EquilibriumBVP(EquilibriumBase):
    """Main class for calculating equilibrium solution


    Inherits analytic parts (gas and dust densities, vertical velocities) from EquilibriumBase, and adds the numerical solution for the horizontal velocities.
    """

    def solve(self, N, neglect_gas_viscosity=False):
        """Solve for the horizontal velocities"""

        # Set up BVP
        self.bvp = EquilibriumSolver(interval=[0, np.inf],
                                     symmetry='even',
                                     basis='Chebychev')

        # Neglect gas viscosity when asked
        a = self.viscous_alpha
        if neglect_gas_viscosity is True:
            a = 0

        # Number of equations: 2 gas + 2 for each dust species
        n_eq = 2 + 2*self.n_dust

        # Solve and store solution for evaluation
        self.sol = self.bvp.solve(N, L=1, n_eq=n_eq,
                                  tau=self.tau,
                                  beta=self.beta(self.tau),
                                  w=self.weights,
                                  mu=self.epsilon,
                                  dlogrhogdz = self.dlogrhogdz,
                                  viscous_alpha=a)

    def evaluate(self, z, k=0):
        """Evaluate solution for horizontal velocities"""

        if self.n_dust == 0:
            if k == 0:
                return 0*z, -self.eta + 0*z, 0*z, 0*z
            else:
                return 0*z, 0*z, 0*z, 0*z

        # Number of equations: 2 gas + 2 for each dust species
        n_eq = 2 + 2*self.n_dust

        # Solutions are symmetric, so can use abs(z)
        u =  self.bvp.evaluate(np.abs(z), self.sol, n_eq=n_eq, k=k)

        if k == 1:
            u = u*(np.sign(z) + (z == 0))

        # Transform to units using c_s = Omega = 1
        u = u*self.eta

        vx = u[0,:]
        vy = u[1,:]
        ux = u[2:self.n_dust+2,:]
        uy = u[2+self.n_dust:,:]

        if k == 0:
            vy = vy - self.eta
            for n in range(0, self.n_dust):
                ux[n,:] -= \
                  2*self.eta*self.tau[n]/(1 + self.tau[n]*self.tau[n])
                uy[n,:] -= self.eta/(1 + self.tau[n]*self.tau[n])

        return vx, vy, ux, uy
