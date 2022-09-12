# -*- coding: utf-8 -*-
"""Module containing DirectSolver class.
"""

import numpy as np
from scipy import sparse

from spectral_solvers.spectral_solvers import EigenValueSolver

from .equilibrium import beta, dust_diffusion_coefficient

class DirectSolver(EigenValueSolver):
    """Eigenvalue solver for stratified PSI.

    """

    def matrixM(self, kx=1, equilibrium=None, neglect_gas_viscosity=False):
        """Calculate matrix we need eigenvalues of

        Args:
            kx: optional, wave number x
            equilibrium: optional, Equilibrium object with background state
            neglect_gas_viscosity: optional, boolean
        """
        n_dust = len(equilibrium.stokes_numbers)

        n_eq = 4 + 4*n_dust
        n_coll = len(self.basis.collocation_points())

        alpha = equilibrium.viscous_alpha
        if neglect_gas_viscosity is True:
            alpha = 0
        bta = beta(equilibrium.stokes_numbers)
        diff = dust_diffusion_coefficient(equilibrium.stokes_numbers,
                                       equilibrium.viscous_alpha)
        shr = 1.5

        # Equilibrium horizontal velocities at collocation points
        vx0, vy0, vz0, ux0, uy0, uz0 = \
            equilibrium.evaluate_velocities(self.z)
        dvx0, dvy0, dvz0, dux0, duy0, duz0 = \
            equilibrium.evaluate_velocities(self.z, k=1)
        d2vx0, d2vy0, d2vz0, d2ux0, d2uy0, d2uz0 = \
            equilibrium.evaluate_velocities(self.z, k=2)

        # rhog'/rhog
        dlog_rhog = equilibrium.eq_rhog.log_deriv(self.z)
        # (rhog'/rhog)'
        d2log_rhog = equilibrium.eq_rhog.second_log_deriv(self.z)
        # sigma'/sigma
        dlog_sigma = equilibrium.eq_sigma.log_deriv(self.z)
        # mu'/mu
        dlog_mu = equilibrium.eq_mu.log_deriv(self.z)

        # dust to gas ratio
        dg_ratio = equilibrium.eq_mu.evaluate(self.z)

        # Full matrix
        ret = np.zeros((n_coll*n_eq, n_coll*n_eq), dtype=np.cdouble)

        indptr = [0]
        indices = []
        data = []

        inv_mat_a = np.eye(n_coll)
        if self.sparse_flag:
            inv_mat_a = np.linalg.inv(self.A)

        # Loop through all equations
        for i in range(0, n_eq):
            indptr.append(0)

            # Loop through all perturbed quantities
            for j in range(0, n_eq):
                # By default, add zeros
                mat_m = np.zeros((n_coll, n_coll), dtype=np.cdouble)

                # Gas continuity equation
                if i == 0:
                    # Gas density perturbation
                    if j == 0:
                        indptr[-1] += 1
                        indices.append(j)

                        f_0 = kx*vx0
                        mat_m = self.construct_m(f_0)
                        data.append(inv_mat_a @ mat_m)
                    # Gas mx perturbation
                    if j == 1:
                        indptr[-1] += 1
                        indices.append(j)

                        mat_m = kx*self.A
                        data.append(inv_mat_a @ mat_m)
                    # Gas mz perturbation
                    if j == 3:
                        indptr[-1] += 1
                        indices.append(j)

                        mat_m = -1j*self.dA
                        data.append(inv_mat_a @ mat_m)

                # Gas momentum x
                if i == 1:
                    # Gas density perturbation
                    if j == 0:
                        indptr[-1] += 1
                        indices.append(j)

                        f_0 = -1j*alpha*(dlog_rhog*dvx0 + d2vx0) + kx
                        for i_d, wgt in enumerate(equilibrium.weights):
                            dvel = ux0[i_d,:]- vx0
                            f_0 -= 1j*dg_ratio[i_d,:]*dvel*wgt
                        mat_m = self.construct_m(f_0)
                        data.append(inv_mat_a @ mat_m)
                    # Gas mx perturbation
                    if j == 1:
                        indptr[-1] += 1
                        indices.append(j)

                        f_0 = kx*vx0 - 4j*alpha*kx*kx/3
                        for i_d in range(0, n_dust):
                            f_0 -= 1j*dg_ratio[i_d,:]*equilibrium.weights[i_d]
                        f_1 = 1j*alpha*dlog_rhog
                        mat_m = self.construct_m(f_0) + \
                          self.construct_m(f_1, k=1) + 1j*alpha*self.ddA
                        data.append(inv_mat_a @ mat_m)
                    # Gas my perturbation
                    if j == 2:
                        indptr[-1] += 1
                        indices.append(j)

                        mat_m = 2j*self.A
                        data.append(inv_mat_a @ mat_m)
                    # Gas mz perturbation
                    if j == 3:
                        indptr[-1] += 1
                        indices.append(j)

                        f_0 = -1j*dvx0 - alpha*kx*dlog_rhog
                        f_1 = -alpha*kx/3
                        mat_m = self.construct_m(f_0) - alpha*kx*self.dA/3
                        data.append(inv_mat_a @ mat_m)
                    # Dust density perturbation
                    if j > 3 and ((j - 4) % 4) == 0:
                        indptr[-1] += 1
                        indices.append(j)

                        i_d = int((j - 4)/4)
                        #f_0 = 1j*(ux0[i_d,:] - vx0)/equilibrium.stokes_numbers[i_d]
                        f_0 = 1j*(ux0[i_d,:] - vx0)*equilibrium.weights[i_d]
                        mat_m = self.construct_m(f_0)
                        data.append(inv_mat_a @ mat_m)
                    # Dust vx perturbation
                    if j > 4 and ((j - 5) % 4) == 0:   # 5, 9, 13, ...
                        indptr[-1] += 1
                        indices.append(j)

                        i_d = int((j - 5)/4)
                        #mat_m = 1j*self.A/equilibrium.stokes_numbers[i_d]
                        mat_m = 1j*self.A*equilibrium.weights[i_d]
                        data.append(inv_mat_a @ mat_m)

                # Gas momentum y
                if i == 2:
                    # Gas density perturbation
                    if j == 0:
                        indptr[-1] += 1
                        indices.append(j)

                        f_0 = -1j*alpha*(dlog_rhog*dvy0 + d2vy0)
                        for i_d, wgt in enumerate(equilibrium.weights):
                            dvel = uy0[i_d,:] - vy0
                            f_0 -= 1j*dg_ratio[i_d,:]*dvel*wgt
                        mat_m = self.construct_m(f_0)
                        data.append(inv_mat_a @ mat_m)
                    # Gas vx perturbation
                    if j == 1:
                        indptr[-1] += 1
                        indices.append(j)

                        mat_m = 1j*(shr - 2)*self.A
                        data.append(inv_mat_a @ mat_m)
                    # Gas vy perturbation
                    if j == 2:
                        indptr[-1] += 1
                        indices.append(j)

                        f_0 = kx*vx0 - 1j*alpha*kx*kx
                        for i_d in range(0, n_dust):
                            f_0 -= 1j*dg_ratio[i_d,:]*equilibrium.weights[i_d]
                        f_1 = 1j*alpha*dlog_rhog
                        mat_m = self.construct_m(f_0) + \
                          self.construct_m(f_1, k=1) + 1j*alpha*self.ddA
                        data.append(inv_mat_a @ mat_m)
                    # Gas vz perturbation
                    if j == 3:
                        indptr[-1] += 1
                        indices.append(j)

                        f_0 = -1j*dvy0
                        mat_m = self.construct_m(f_0)
                        data.append(inv_mat_a @ mat_m)
                    # Dust density perturbation
                    if j > 3 and ((j - 4) % 4) == 0:
                        indptr[-1] += 1
                        indices.append(j)

                        i_d = int((j - 4)/4)
                        #f_0 = 1j*(uy0[i_d,:] - vy0)/equilibrium.stokes_numbers[i_d]
                        f_0 = 1j*(uy0[i_d,:] - vy0)*equilibrium.weights[i_d]
                        mat_m = self.construct_m(f_0)
                        data.append(inv_mat_a @ mat_m)
                    # Dust vy perturbation
                    if j > 5 and ((j - 6) % 4) == 0:
                        indptr[-1] += 1
                        indices.append(j)

                        i_d = int((j - 6)/4)
                        #mat_m = 1j*self.A/equilibrium.stokes_numbers[i_d]
                        mat_m = 1j*self.A*equilibrium.weights[i_d]
                        data.append(inv_mat_a @ mat_m)

                # Gas momentum z
                if i == 3:
                    # Gas density perturbation
                    if j == 0:
                        indptr[-1] += 1
                        indices.append(j)

                        f_0 = 1j*dlog_rhog
                        for i_d in range(0, n_dust):
                            wgt = equilibrium.weights[i_d]
                            f_0 += 1j*dg_ratio[i_d,:]*bta[i_d]*self.z*wgt
                        mat_m = self.construct_m(f_0) - 1j*self.dA
                        data.append(inv_mat_a @ mat_m)
                    # Gas vx perturbation
                    if j == 1:
                        indptr[-1] += 1
                        indices.append(j)

                        f_0 = 2*alpha*kx*dlog_rhog/3
                        mat_m = self.construct_m(f_0) -alpha*kx*self.dA/3
                        data.append(inv_mat_a @ mat_m)
                    # Gas vz perturbation
                    if j == 3:
                        indptr[-1] += 1
                        indices.append(j)

                        f_0 = kx*vx0 - 1j*alpha*kx*kx
                        for i_d in range(0, n_dust):
                            f_0 -= 1j*dg_ratio[i_d,:]*equilibrium.weights[i_d]
                        f_1 = 4j*alpha*dlog_rhog/3
                        mat_m = self.construct_m(f_0) + \
                          self.construct_m(f_1, k=1) + 4j*alpha*self.ddA/3
                        data.append(inv_mat_a @ mat_m)
                    # Dust density perturbation
                    if j > 3 and ((j - 4) % 4) == 0:
                        indptr[-1] += 1
                        indices.append(j)

                        i_d = int((j - 4)/4)
                        #f_0 = -1j*bta[i_d]*self.z/equilibrium.stokes_numbers[i_d]
                        f_0 = -1j*bta[i_d]*self.z*equilibrium.weights[i_d]
                        mat_m = self.construct_m(f_0)
                        data.append(inv_mat_a @ mat_m)
                    # Dust uz perturbation
                    if j > 6 and ((j - 7) % 4) == 0:
                        indptr[-1] += 1
                        indices.append(j)

                        i_d = int((j - 7)/4)
                        #mat_m = 1j*self.A/equilibrium.stokes_numbers[i_d]
                        mat_m = 1j*self.A*equilibrium.weights[i_d]
                        data.append(inv_mat_a @ mat_m)

                # Dust continuity
                if i > 3 and ((i - 4) % 4) == 0:
                    i_d = int((i - 4)/4)

                    # Gas density perturbation
                    if j == 0:
                        indptr[-1] += 1
                        indices.append(j)

                        f_0 = 1j*diff[i_d]*dg_ratio[i_d,:]*\
                            (kx*kx + dlog_mu[i_d,:]*dlog_rhog + d2log_rhog)
                        f_1 = -1j*diff[i_d]*dg_ratio[i_d,:]*(dlog_mu[i_d,:] - dlog_rhog)
                        f_2 = -1j*diff[i_d]*dg_ratio[i_d,:]
                        mat_m = self.construct_m(f_0) + \
                          self.construct_m(f_1, k=1) + self.construct_m(f_2, k=2)
                        data.append(inv_mat_a @ mat_m)
                    # Dust density perturbation
                    if j == i:
                        indptr[-1] += 1
                        indices.append(j)

                        f_0 = kx*ux0[i_d,:] + 1j*bta[i_d] - \
                          1j*diff[i_d]*(kx*kx + d2log_rhog)
                        f_1 = -1j*diff[i_d]*dlog_rhog + 1j*bta[i_d]*self.z
                        mat_m = self.construct_m(f_0) + self.construct_m(f_1, k=1) + \
                          1j*diff[i_d]*self.ddA
                        data.append(inv_mat_a @ mat_m)
                    # Dust ux perturbation
                    if j == i + 1:
                        indptr[-1] += 1
                        indices.append(j)

                        mat_m = kx*self.A
                        data.append(inv_mat_a @ mat_m)
                    # Dust uz perturbation
                    if j == i + 3:
                        indptr[-1] += 1
                        indices.append(j)

                        mat_m = -1j*self.dA
                        data.append(inv_mat_a @ mat_m)

                # Dust momentum x
                if i > 4 and ((i - 5) % 4) == 0:
                    i_d = int((i - 5)/4)

                    # Gas vx perturbation
                    if j == 1:
                        indptr[-1] += 1
                        indices.append(j)

                        f_0 = 1j*dg_ratio[i_d,:]/equilibrium.stokes_numbers[i_d]
                        mat_m = self.construct_m(f_0)
                        data.append(inv_mat_a @ mat_m)
                    # Dust ux perturbation
                    if j == i:
                        indptr[-1] += 1
                        indices.append(j)

                        f_0 = kx*ux0[i_d,:] - 1j/equilibrium.stokes_numbers[i_d] \
                          -1j*bta[i_d]*self.z*dlog_sigma[i_d,:]
                        f_1 = 1j*bta[i_d]*self.z
                        mat_m = self.construct_m(f_0) + self.construct_m(f_1, k=1)
                        data.append(inv_mat_a @ mat_m)
                    # Dust uy perturbation
                    if j == i + 1:
                        indptr[-1] += 1
                        indices.append(j)

                        mat_m = 2j*self.A
                        data.append(inv_mat_a @ mat_m)
                    # Dust uz perturbation
                    if j == i + 2:
                        indptr[-1] += 1
                        indices.append(j)

                        f_0 = -1j*dux0[i_d,:]
                        mat_m = self.construct_m(f_0)
                        data.append(inv_mat_a @ mat_m)

                # Dust momentum y
                if i > 5 and ((i - 6) % 4) == 0:
                    i_d = int((i - 6)/4)

                    # Gas vy perturbation
                    if j == 2:
                        indptr[-1] += 1
                        indices.append(j)

                        f_0 = 1j*dg_ratio[i_d,:]/equilibrium.stokes_numbers[i_d]
                        mat_m = self.construct_m(f_0)
                        data.append(inv_mat_a @ mat_m)
                    # Dust ux perturbation
                    if j == i - 1:
                        indptr[-1] += 1
                        indices.append(j)

                        mat_m = 1j*(shr - 2)*self.A
                        data.append(inv_mat_a @ mat_m)
                    # Dust uy perturbation
                    if j == i:
                        indptr[-1] += 1
                        indices.append(j)

                        f_0 = kx*ux0[i_d,:] - \
                          1j/equilibrium.stokes_numbers[i_d] - \
                          1j*bta[i_d]*self.z*dlog_sigma[i_d,:]
                        f_1 = 1j*bta[i_d]*self.z
                        mat_m = self.construct_m(f_0) + self.construct_m(f_1, k=1)
                        data.append(inv_mat_a @ mat_m)
                    # Dust uz perturbation
                    if j == i + 1:
                        indptr[-1] += 1
                        indices.append(j)

                        f_0 = -1j*duy0[i_d,:]
                        mat_m = self.construct_m(f_0)
                        data.append(inv_mat_a @ mat_m)

                # Dust momentum z
                if i > 6 and ((i - 7) % 4) == 0:
                    i_d = int((i - 7)/4)

                    # Gas vz perturbation
                    if j == 3:
                        indptr[-1] += 1
                        indices.append(j)

                        f_0 = 1j*dg_ratio[i_d,:]/equilibrium.stokes_numbers[i_d]
                        mat_m = self.construct_m(f_0)
                        data.append(inv_mat_a @ mat_m)
                    # Dust uz perturbation
                    if j == i:
                        indptr[-1] += 1
                        indices.append(j)

                        f_0 = kx*ux0[i_d,:] - \
                          1j/equilibrium.stokes_numbers[i_d] + 1j*bta[i_d] - \
                          1j*bta[i_d]*self.z*dlog_sigma[i_d,:]
                        f_1 = 1j*bta[i_d]*self.z
                        mat_m = self.construct_m(f_0) + self.construct_m(f_1, k=1)
                        data.append(inv_mat_a @ mat_m)

                ret[i*n_coll:(i+1)*n_coll, j*n_coll:(j+1)*n_coll] = mat_m

        data = np.asarray(data)
        indices = np.asarray(indices)
        indptr = np.cumsum(np.asarray(indptr))

        ret = sparse.bsr_matrix((data, indices, indptr), shape=(n_eq*n_coll, n_eq*n_coll))

        if not self.sparse_flag:
            ret = ret.todense()

        return ret
