# -*- coding: utf-8 -*-
"""Module dealing with computing equilibrium solution.
"""

import numpy as np

from scipy import integrate
from scipy.special import roots_legendre
from scipy.optimize import fsolve

from spectral_solvers.spectral_solvers import BoundaryValueProblemSolver

def dust_diffusion_coefficient(stokes, alpha):
    """Dust diffusion coefficient

    Args:
        stokes (ndarray): Stokes number
        alpha: Viscous alpha on which dust diffusion is based.

    Returns:
        dust diffusion coefficient, in units of c*c/Omega
    """
    return (1 + stokes + 4*stokes**2)*alpha/(1 + stokes)**2

def beta(stokes):
    """Vertical dust velocity is -beta*z. Valid for Stokes numbers < 1/2.

    Args:
        stokes (ndarray): Stokes number

    Returns:
        coefficient of vertical dust velocity, in units of Omega
    """
    return 0.5*(1/stokes - np.sqrt(1/stokes/stokes - 4))

class EquilibriumProfile():
    """Base for equilibrium state

    In a stratified shearing box, equilibrium profiles (density, velocity)
    depend on the vertical coordinate z.

    Args:
        metallicity: desired metallicity
        stokes_density: StokesDensity object, encoding the size distribution
        stokes_numbers: minimum and maximum Stokes number considered
        viscous_alpha: gas alpha viscosity
        weights: Gauss-Legendre weights in size space
    """
    def __init__(self, metallicity, stokes_density, stokes_numbers,
                 viscous_alpha, weights):
        self.metallicity = metallicity
        self.stokes_density = stokes_density
        self.alpha = viscous_alpha
        self.stokes_numbers = stokes_numbers
        self.weights = weights

    def make_1d(self, z_coord):
        """Make z_coord an array

        This function helps to deal with both scalar and array input, forcing
        z_coord to be a 1D array.

        Args:
            z_coord: vertical coordinate, can be scalar or ndarray

        Returns:
            z_coord: vertical coordinate, 1D array
            scalar_input: boolean, true if riginal input was a scalar
            original_shape: original shape of z_coord
        """
        z_coord = np.asarray(z_coord)
        scalar_input = False
        if z_coord.ndim == 0:
            z_coord = z_coord[None]  # Makes z_coord 1D
            scalar_input = True
            original_shape = None
        else:
            original_shape = np.shape(z_coord)
            z_coord = np.ravel(z_coord)

        return z_coord, scalar_input, original_shape

    def return_to_original_shape(self, res, scalar_input, original_shape):
        """Return z_coord to original shape

        Args:
            res: array to convert to original shape
            scalar_input: boolean, true if riginal input was a scalar
            original_shape: original shape of z_coord

        Returns:
            res in original shape of z_coord
        """
        if scalar_input:
            return np.squeeze(res)
        return np.reshape(res, original_shape)

class EquilibriumDustToGasRatio(EquilibriumProfile):
    """Dust to gas ratio equilibrium profile

    When created, the mid plane dust to gas ratio is set equal to 100x the
    metallicity. A more accurate calculation should be done to set midplane_dg
    to a value that gives the desired metallicity.

    Args:
        metallicity: desired metallicity
        stokes_density: StokesDensity object, encoding the size distribution
        stokes_numbers: minimum and maximum Stokes number considered
        viscous_alpha: gas alpha viscosity
        weights: Gauss-Legendre weights in size space
    """
    def __init__(self, metallicity, stokes_density, stokes_numbers,
                 viscous_alpha, weights):
        EquilibriumProfile.__init__(self, metallicity, stokes_density,
                                    stokes_numbers, viscous_alpha, weights)

        # Approximate: should be reset later.
        self.midplane_dg = \
            np.sqrt(stokes_numbers[-1]/viscous_alpha)*self.metallicity

    def evaluate(self, z_coord, stokes_number=None):
        """Dust to gas ratio

        Evaluate d/g ratio, either for all Stokes numbers (if stokes_number is
        None) or at specific Stokes numbers.

        Args:
            z_coord: vertical coordinate where dust to gas ratio is required
            stokes_number: optional, Stokes numbers for d/g

        Returns:
            dust to gas ratio at required Stokes numbers
        """
        if stokes_number is not None:
            # Evaluate at specific stokes number
            fac = 0.5*beta(stokes_number)*z_coord*z_coord
            fac = fac/dust_diffusion_coefficient(stokes_number, self.alpha)
            return \
              self.midplane_dg*self.stokes_density.sigma(stokes_number)*np.exp(-fac)

        # All Stokes numbers
        ret = np.zeros((len(self.stokes_numbers), len(z_coord)))

        for i, stk in enumerate(self.stokes_numbers):
            ret[i, :] = self.midplane_dg*self.stokes_density.sigma(stk)* \
              np.exp(-0.5*beta(stk)*z_coord*z_coord/\
                    dust_diffusion_coefficient(stk, self.alpha))

        return ret

    def log_deriv(self, z_coord):
        """Logarithmic derivative dust-to-gas ratio for all species

        Args:
            z_coord: vertical coordinate

        Returns:
            logarithmic derivative of dust-to-gas ratio

        """
        ret = np.zeros((len(self.stokes_numbers), len(z_coord)))

        for i, stk in enumerate(self.stokes_numbers):
            ret[i, :] = \
                -beta(stk)*z_coord/dust_diffusion_coefficient(stk, self.alpha)

        return ret

    def second_log_deriv(self, z_coord):
        '''Second derivative of dust-to-gas ratio/ mu for all species

        Args:
            z_coord: vertical coordinate

        Returns:
            second derivative of dust-to-gas-ratio, divided by the
            dust-to-gas ratio
        '''
        ret = np.zeros((len(self.stokes_numbers), len(z_coord)))

        for i, stk in enumerate(self.stokes_numbers):
            diff = dust_diffusion_coefficient(stk, self.alpha)
            ret[i, :] = beta(stk)*(beta(stk)*z_coord*z_coord/diff - 1)/diff

        return ret


class EquilibriumGasDensity(EquilibriumProfile):
    """Gas density equilibrium profile

    Requires a valid instance of EquilibriumDustToGasRatio.

    Args:
        metallicity: desired metallicity
        stokes_density: StokesDensity object encoding the size distribution
        stokes_numbers: minimum and maximum Stokes number
        viscous_alpha: gas alpha viscosity
        weights: Gauss-Legendre weights in size space
        eq_dust_to_gas_ratio: equilibrium profile d/g ratio
    """
    def __init__(self, metallicity, stokes_density, stokes_numbers,
                 viscous_alpha, weights, eq_dust_to_gas_ratio):
        EquilibriumProfile.__init__(self, metallicity, stokes_density,
                                    stokes_numbers, viscous_alpha, weights)

        self.dust_to_gas_ratio = eq_dust_to_gas_ratio

    def evaluate(self, z_coord):
        """Gas density

        Args:
            z_coord: vertical coordinate

        Returns:
            gas density, normalized to unity in the mid plane
        """
        z_coord, scalar_input, original_shape = self.make_1d(z_coord)

        # Dust-free limit
        ret = -0.5*z_coord*z_coord

        # Eq. d/g ratio
        mu1 = self.dust_to_gas_ratio.evaluate(z_coord)
        # Eq. d/g ratio in mid plane
        mu0 = self.dust_to_gas_ratio.evaluate(np.asarray([0.0]))

        # Perform integral using Legendre quadrature
        for i, stk in enumerate(self.stokes_numbers):
            ret = ret + \
              dust_diffusion_coefficient(stk, self.alpha)*\
              (mu1[i,:] - mu0[i,:])*self.weights[i]

        return self.return_to_original_shape(np.exp(ret),
                                             scalar_input,
                                             original_shape)

    def log_deriv(self, z_coord):
        """Logarithmic z derivative of gas density

        Args:
            z_coord: vertical coordinate

        Returns:
            logarithmic derivative of gas density
        """
        ret = -z_coord

        # Derivative of d/g ratio
        dmu_dz = \
          self.dust_to_gas_ratio.evaluate(z_coord)*\
          self.dust_to_gas_ratio.log_deriv(z_coord)

        # Perform integral using Gauss-Legendre quadrature
        for i, stk in enumerate(self.stokes_numbers):
            ret = ret + \
              dust_diffusion_coefficient(stk, self.alpha)*\
              dmu_dz[i,:]*self.weights[i]

        return ret

    def second_log_deriv(self, z_coord):
        """ (rhog'/rhog)'

        Args:
            z_coord: vertical coordinate

        Returns:
            derivative of logarithmic derivative of gas density
        """
        ret = -1 + z_coord - z_coord

        mu1 = self.dust_to_gas_ratio.evaluate(z_coord)
        dmu = self.dust_to_gas_ratio.log_deriv(z_coord)

        for i, stk in enumerate(self.stokes_numbers):
            ret -= \
              self.weights[i]*beta(stk)*mu1[i,:]*(1 + z_coord*dmu[i,:])

        return ret

    def integrate(self, height=10):
        """Gas surface density

        Args:
            height: optional, integration bounds in terms of scale height

        Returns:
            gas surface density
        """
        res = 2*integrate.quad(self.evaluate, 0, height)[0]
        return res

class EquilibriumStokesDensity(EquilibriumProfile):
    """Stokes density equilibrium profile

    Requires valid instances of EquilibriumDustToGasRatio and
    EquilibriumGasDensity.

    Args:
        metallicity: desired metallicity
        stokes_density: StokesDensity object encoding the size distribution
        stokes_numbers: minimum and maximum Stokes number
        viscous_alpha: gas alpha viscosity
        weights: Gauss-Legendre weights in size space
        eq_dust_to_gas_ratio: equilibrium profile gas density
    """
    def __init__(self, metallicity, stokes_density, stokes_numbers,
                 viscous_alpha, weights, eq_gas_density):
        EquilibriumProfile.__init__(self, metallicity, stokes_density,
                                    stokes_numbers, viscous_alpha, weights)

        self.gas_density = eq_gas_density
        self.dust_to_gas_ratio = eq_gas_density.dust_to_gas_ratio

    def evaluate(self, z_coord, stokes_number=None):
        """Stokes density

        Args:
            z_coord: vertical coordinate
            stokes_number: optional, evaluate at specific Stokes numbers

        Returns:
            Stokes density
        """
        return self.gas_density.evaluate(z_coord)*\
          self.dust_to_gas_ratio.evaluate(z_coord, stokes_number=stokes_number)

    def log_deriv(self, z_coord):
        '''Logarithmic derivative size density for all species

        Args:
            z_coord: vertical coordinate

        Returns:
            logarithmic derivative Stokes density
        '''
        ret = self.dust_to_gas_ratio.log_deriv(z_coord)

        for i in range(0, len(self.stokes_numbers)):
            ret[i, :] += self.gas_density.log_deriv(z_coord)

        return ret

    def integrate_stokes(self, z_coord):
        """Dust density, integrated over Stokes number

        Args:
            z_coord: vertical coordinate

        Returns:
            dust density
        """
        z_coord, scalar_input, original_shape = self.make_1d(z_coord)
        res = 0*z_coord

        # Integrate over Stokes number
        for i in range(0, len(z_coord)):

            def stokes_dens(stk):
                return self.dust_to_gas_ratio.midplane_dg*\
                  self.gas_density.evaluate(z_coord[i])\
                  *np.exp(-0.5*beta(stk)*z_coord[i]*z_coord[i]/\
                  dust_diffusion_coefficient(stk, self.alpha))

            #f = lambda x: self.dust_to_gas_ratio.midplane_dg*\
            #    self.gas_density.evaluate(z_coord[i])\
            #    *np.exp(-0.5*beta(x)*z_coord[i]*z_coord[i]/\
            #        dust_diffusion_coefficient(x, self.alpha))
            res[i] = self.stokes_density.integrate(stokes_dens)

        return self.return_to_original_shape(res,
                                             scalar_input,
                                             original_shape)

    def integrate(self, height=10):
        """Dust surface density

        Args:
            height: optional, integration bounds in terms of gas scale height

        Returns:
            dust surface density
        """
        res = 2*integrate.quad(self.integrate_stokes, 0, height)[0]
        return res

class EquilibriumVelocities(EquilibriumProfile):
    """Equilibrium velocity profile

    Args:
        metallicity: desired metallicity
        stokes_density: StokesDensity object encoding the size distribution
        stokes_numbers: minimum and maximum Stokes number
        viscous_alpha: gas alpha viscosity
        weights: Gauss-Legendre weights in size space
    """

    def __init__(self, metallicity, stokes_density, stokes_numbers,
                 viscous_alpha, weights):
        EquilibriumProfile.__init__(self, metallicity, stokes_density,
                                    stokes_numbers, viscous_alpha, weights)

        self.bvp = None
        self.sol = None

    def solve(self, n_coll, neglect_viscosity, dust_to_gas_ratio, gas_density):
        """Solve numerically for equilibrium velocities

        Sets self.sol and self.bvp for future use.

        Args:
            n_coll: number of collocation points in z
            neglect_viscosity: boolean whether to neglect gas viscosity
            dust_to_gas_ratio: equilibrium dust to gas ratio profile
            gas_density: equilibrium gas density profile

        Returns:
            Nothing, but self.sol and self.bvp are set for future use
        """
        # Set up BVP
        self.bvp = EquilibriumSolver(interval=[0, np.inf],
                                     symmetry='even',
                                     basis='Chebychev')

        # Neglect gas viscosity when asked
        alpha = self.alpha
        if neglect_viscosity is True:
            alpha = 0

        # Number of equations: 2 gas + 2 for each dust species
        n_eq = 2 + 2*len(self.stokes_numbers)

        # Solve and store solution for evaluation
        self.sol = self.bvp.solve(n_coll, L=1, n_eq=n_eq,
                                  tau=self.stokes_numbers,
                                  w=self.weights,
                                  mu=dust_to_gas_ratio.evaluate,
                                  dlogrhogdz = gas_density.log_deriv,
                                  viscous_alpha=alpha)

    def evaluate(self, z_coord, k=0, eta=0.05):
        """Evaluate velocities

        Args:
            z_coord: vertical coordinate
            k: optional, order of derivative to take
            eta: optional, radial pressure gradient parameter
        """
        # Number of dust species
        n_dust = len(self.stokes_numbers)

        # Gas-only case
        if n_dust == 0:
            if k == 0:
                return 0*z_coord, -eta + 0*z_coord, 0*z_coord, 0*z_coord, 0*z_coord, 0*z_coord
            return 0*z_coord, 0*z_coord, 0*z_coord, 0*z_coord, 0*z_coord, 0*z_coord

        # Number of equations: 2 gas + 2 for each dust species
        n_eq = 2 + 2*n_dust

        # Solutions are symmetric, so can use abs(z_coord)
        sol =  self.bvp.evaluate(np.abs(z_coord), self.sol, n_eq=n_eq, k=k)

        if k == 1:
            sol = sol*(np.sign(z_coord) + (z_coord == 0))

        # Transform to units using c_s = Omega = 1
        sol = sol*eta

        gas_vx = sol[0,:]
        gas_vy = sol[1,:]
        gas_vz = 0.0*gas_vx
        dust_ux = sol[2:n_dust + 2,:]
        dust_uy = sol[2 + n_dust:,:]
        dust_uz = 0.0*dust_ux

        if k == 0:
            gas_vy = gas_vy - eta
            for i, stk in enumerate(self.stokes_numbers):
                dust_ux[i,:] -= 2*eta*stk/(1 + stk**2)
                dust_uy[i,:] -= eta/(1 + stk**2)
                dust_uz[i,:] = -beta(stk)*z_coord

        if k == 1:
            for i, stk in enumerate(self.stokes_numbers):
                dust_uz[i,:] = -beta(stk)

        return gas_vx, gas_vy, gas_vz, dust_ux, dust_uy, dust_uz

class EquilibriumSolver(BoundaryValueProblemSolver):
    """Solve BVP for horizontal velocities

    Horizontal velocities are found from a BVP on an infinite interval.
    Spectral collocation transforms the BVP into a matrix equation M*x = S,
    where M is a matrix (obtained from differential operators), S is the
    source term, and x is the vector of unknowns. The class
    BoundaryValueProblemSolver contains all the necessary machinery to
    solve this; we only need to define M and S.
    """

    def vectorS(self, tau=None, w=None, mu=None,
                dlogrhogdz=None, viscous_alpha=0):
        """Calculate source terms for BVP

        Note that vectorS and matrixM expect the same parameters, even though
        not all of them are used in each function.

        Args:
            tau: Stokes numbers
            w: Gauss-Legendre weights for Stokes number integration
            mu: dust to gas ratio(z, tau)
            dlogrhogdz: logarithmic derivative of gas density
            viscous_alpha: alpha viscosity parameter
        """

        # Number of equations: 2 gas + 2 for every dust species
        n_eq = 2 + 2*len(tau)

        # Number of collocation points
        n_coll = len(self.basis.collocation_points())

        # Full vector: N entries per equation
        ret = np.zeros((n_coll*n_eq), dtype=np.double)

        # Only gas equations have a source
        for i, stk in enumerate(tau):
            ret[0*n_coll:(0+1)*n_coll] += \
                2*mu(self.z, stk)*w[i]*stk/(1 + stk**2)
            ret[1*n_coll:(1+1)*n_coll] -= \
                mu(self.z, stk)*w[i]*stk**2/(1 + stk**2)

        return ret

    def matrixM(self, tau=None, w=None, mu=None,
                dlogrhogdz=None, viscous_alpha=0):
        """Calculate matrix M for BVP M*x = S

        If viscous_alpha is zero we are neglecting gas viscosity.

        Args:
            tau: Stokes numbers
            w: Gauss-Legendre weights for Stokes number integration
            mu: dust to gas ratio(z, tau)
            dlogrhogdz: logarithmic derivative of gas density
            viscous_alpha: gas viscosity
        """

        # Number of dust species
        n_dust = len(tau)

        beta_val = beta(tau)

        # Number of equations: 2 gas + 2 for every dust species
        n_eq = 2 + 2*n_dust

        # Number of collocation points
        n_coll = len(self.basis.collocation_points())

        # Full matrix
        ret = np.zeros((n_coll*n_eq, n_coll*n_eq), dtype=np.double)

        # Loop through all equations
        for i in range(0, n_eq):
            # Loop through all velocities
            for j in range(0, n_eq):
                # By default, add zeros
                mat = np.zeros((n_coll, n_coll), dtype=np.double)

                # vgx equation
                if i == 0:
                    if j == 0:
                        # vgx
                        f_0 = 0*self.z
                        for i_d in range(0, n_dust):
                            f_0 -= mu(self.z, tau[i_d])*w[i_d]
                        f_1 = viscous_alpha*dlogrhogdz(self.z)
                        mat = viscous_alpha*self.ddA + \
                          self.construct_m(f_1, k=1) + self.construct_m(f_0)
                    if j == 1:
                        # vgy
                        mat = 2*self.A
                    if 1 < j < 2 + n_dust:
                        # u_x
                        f_0 = w[j-2]*mu(self.z, tau[j-2])
                        mat = self.construct_m(f_0)

                # vgy equation
                if i == 1:
                    if j == 0:
                        # vgx
                        mat = -0.5*self.A
                    if j == 1:
                        # vgy
                        f_0 = 0*self.z
                        for i_d in range(0, n_dust):
                            f_0 -= mu(self.z, tau[i_d])*w[i_d]
                        f_1 = viscous_alpha*dlogrhogdz(self.z)
                        mat = viscous_alpha*self.ddA + \
                          self.construct_m(f_1, k=1) + self.construct_m(f_0)
                    if j >= 2 + n_dust:
                        # u_y
                        f_0 = w[j-2-n_dust]*mu(self.z, tau[j-2-n_dust])
                        mat = self.construct_m(f_0)

                # ux equations
                if 1 < i < 2 + n_dust:
                    if j == 0:
                        # vgx
                        mat = self.A/tau[i-2]
                    if j == i:
                        # u_x
                        f_1 = beta_val[j-2]*self.z
                        mat = -self.A/tau[j-2] + self.construct_m(f_1, k=1)
                    #if j >= 2 + n_dust:
                    if j == i + n_dust:
                        # u_y
                        mat = 2*self.A

                # uy equations
                if i >= 2 + n_dust:
                    if j == 1:
                        # vgy
                        mat = self.A/tau[i-2-n_dust]
                    if j == i - n_dust:
                        # u_x
                        mat = -0.5*self.A
                    if j == i:
                        # u_y
                        f_1 = beta_val[i-2-n_dust]*self.z
                        mat = -self.A/tau[i-2-n_dust] + self.construct_m(f_1, k=1)


                ret[i*n_coll:(i+1)*n_coll, j*n_coll:(j+1)*n_coll] = mat

        return ret

class Equilibrium():
    """Class holding the equilibrium solution for a polydisperse dusty gas.

    All solutions are horizontally uniform (except for the background shear).
    Solutions in the vertical direction can be obtained analytically for the
    densities and vertical velocities. Horizontal velocities must be obtained
    numerically. Units: time in 1/Omega, length in c/Omega, gas density is
    unity in mid plane.

    Args:
        metallicity: desired ratio of dust and gas surface densities
        stokes_density: StokesDensity object, encoding the size distribution
        n_dust: number of dust collocation points
        viscous_alpha: optional, gas viscosity parameter
    """
    def __init__(self,
                 metallicity,
                 stokes_density,
                 n_dust,
                 viscous_alpha=1.0e-6):

        # Check if parameters are sensible
        if viscous_alpha <= 0.0:
            raise ValueError('Need viscous_alpha to be positive!')

        self.stokes_density = stokes_density
        self.viscous_alpha = viscous_alpha

        self.n_dust = n_dust

        # Monodisperse setup for n_dust=1
        x_coll = np.asarray([1.0])
        self.weights = np.asarray([1.0/self.stokes_density.stokes_max])

        # Minimum and maximum stopping time in size distribution
        st_min = self.stokes_density.stokes_max   # Monodisperse for n_dust=1
        st_max = self.stokes_density.stokes_max

        # Polydisperse setup
        if self.n_dust > 1:
            st_min = self.stokes_density.stokes_min
            x_coll, self.weights = roots_legendre(self.n_dust)

            # Absorb scale factor for integrals into weights
            self.weights = self.weights*0.5*np.log(st_max/st_min)

        # Gauss-Legendre nodes in terms of stopping time
        self.stokes_numbers = st_min*np.power(st_max/st_min, 0.5*(x_coll + 1.0))

        if self.n_dust == 0:
            self.stokes_numbers = []
            self.weights = []

        # Equilibrium dust to gas ratio
        self.eq_mu = EquilibriumDustToGasRatio(metallicity,
                                               self.stokes_density,
                                               self.stokes_numbers,
                                               self.viscous_alpha,
                                               self.weights)
        # Equilibrium gas density
        self.eq_rhog = EquilibriumGasDensity(metallicity,
                                             self.stokes_density,
                                             self.stokes_numbers,
                                             self.viscous_alpha,
                                             self.weights,
                                             self.eq_mu)
        # Equilibrium Stokes density
        self.eq_sigma = EquilibriumStokesDensity(metallicity,
                                                 self.stokes_density,
                                                 self.stokes_numbers,
                                                 self.viscous_alpha,
                                                 self.weights,
                                                 self.eq_rhog)
        # Equilibrium velocities
        self.eq_vel = EquilibriumVelocities(metallicity,
                                            self.stokes_density,
                                            self.stokes_numbers,
                                            self.viscous_alpha,
                                            self.weights)

    def metallicity(self, midplane_dg=None):
        """Metallicity = dust to gas ratio (surface densities)

        Args:
            midplane_dg: optional, sets midplane dg ratio before calculating

        Returns:
            ratio of dust surface density to gas surface density
        """
        if midplane_dg is not None:
            self.eq_mu.midplane_dg = midplane_dg

        res = self.eq_sigma.integrate()/self.eq_rhog.integrate()
        return res

    def set_metallicity(self, desired_metallicity):
        """Attempt to set metallicity

        Calculate which midplane dust to gas ratio corresponds to the given
        metallicity. This value of the dust to gas ratio is then fed into the
        equilibrium density profiles.

        Args:
            desired_metallicity: desired metallicity

        Returns:
            nothing, but sets the midplane dust to gas ratio
        """

        # Minimize metallicity - Z
        res = fsolve(lambda x: self.metallicity(midplane_dg=x) - \
                     desired_metallicity, self.eq_mu.midplane_dg)[0]

        # Set dust to gas ratio to required value
        self.eq_mu.midplane_dg = res

    def get_state(self, z_coord):
        """Get full equilibrium state

        Args:
            z_coord: vertical coordinate

        Returns:
            gas density, Stokes density, dust to gas ratio, dust density
            (integrated over Stokes number), gas velocities (3x) and dust
            velocities (3x)
        """
        rhog = self.eq_rhog.evaluate(z_coord)
        sigma = self.eq_sigma.evaluate(z_coord)
        dg_ratio = self.eq_mu.evaluate(z_coord)
        dust_rho = self.eq_sigma.integrate_stokes(z_coord)
        gas_vx, gas_vy, gas_vz, dust_ux, dust_uy, dust_uz = \
            self.evaluate_velocities(z_coord)

        return rhog, sigma, dg_ratio, dust_rho, \
            gas_vx, gas_vy, gas_vz, dust_ux, dust_uy, dust_uz

    def solve_horizontal_velocities(self, n_coll, neglect_gas_viscosity=False):
        """Solve for the horizontal velocities

        Args:
            n_coll: number of collocation points in z to use
            neglect_gas_viscosity: optional, boolean specifying whether to
            neglect gas viscosity in the calculation of the velocities
        """
        self.eq_vel.solve(n_coll, neglect_gas_viscosity,
                          self.eq_mu, self.eq_rhog)

    def evaluate_velocities(self, z_coord, k=0):
        """Evaluate solution for velocities

        Args:
            z_coord: vertical coordinate
            k: optional, order of derivative to compute (0, 1 or 2)
        """

        gas_vx, gas_vy, gas_vz, dust_ux, dust_uy, dust_uz = \
            self.eq_vel.evaluate(z_coord, k=k)

        return gas_vx, gas_vy, gas_vz, dust_ux, dust_uy, dust_uz
