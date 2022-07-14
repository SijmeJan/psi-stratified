import numpy as np
import scipy.sparse as sparse

from spectral_solvers.spectral_solvers import EigenValueSolver


class HermiteSolver(EigenValueSolver):
    def matrixM(self):
        f1 = self.z
        M = self.construct_m(f1, k=1) + self.ddA
        return M

class LaguerreEigenValueSolver(EigenValueSolver):
    def matrixM(self):
        f2 = self.z
        f0 = 1 - 0.25*self.z
        M = self.construct_m(f2, k=2) + self.construct_m(f0)
        return M

class FourierSolverMulti(EigenValueSolver):
    def matrixM(self):
        n_eq = 2
        N = len(self.basis.collocation_points())

        # Full matrix
        ret = np.zeros((N*n_eq, N*n_eq), dtype=np.cdouble)

        # Loop through all equations
        for i in range(0, n_eq):
            # Loop through all perturbed quantities
            for j in range(0, n_eq):
                # By default, add zeros
                M = np.zeros((N, N), dtype=np.cdouble)

                if i == j:
                    M = self.ddA

                ret[i*N:(i+1)*N, j*N:(j+1)*N] = M

        return ret

class HermiteSolverMulti(EigenValueSolver):
    def matrixM(self, kx=1):
        n_eq = 4
        N = len(self.basis.collocation_points())

        # Shear parameter
        S = 1.5

        #chi = self.z
        #chi_prime = 1 + self.z - self.z
        #chi_prime2_over_chi = 0 + self.z - self.z
        #z_over_chi = 1 + self.z - self.z
        #z_chi_prime_over_chi = 1 + self.z - self.z
        chi = 1 + self.z - self.z
        chi_prime = 0 + self.z - self.z
        chi_prime2_over_chi = 0 + self.z - self.z
        z_over_chi = self.z
        z_chi_prime_over_chi = 0 + self.z - self.z

        # Full matrix
        ret = np.zeros((N*n_eq, N*n_eq), dtype=np.cdouble)

        # Loop through all equations
        for i in range(0, n_eq):
            # Loop through all perturbed quantities
            for j in range(0, n_eq):
                # By default, add zeros
                M = np.zeros((N, N), dtype=np.cdouble)

                # Gas continuity
                if i == 0:
                    # Momentum x perturbation
                    if j == 1:
                        M = kx*self.A
                    # Momentum z perturbation
                    if j == 3:
                        f0 = -1j*chi_prime
                        f1 = -1j*chi
                        M = self.construct_m(f0) + self.construct_m(f1, k=1)

                # Momentum x equation
                if i == 1:
                    # Gas density perturbation
                    if j == 0:
                        M = kx*self.A
                    # Momentum y perturbation
                    if j == 2:
                        M = 2j*self.A
                # Momentum y equation
                if i == 2:
                    # Momentum x perturbation
                    if j == 1:
                        M = -1j*(2 - S)*self.A
                # Momentum z equation
                if i == 3:
                    # Gas density perturbation
                    if j == 0:
                        f0 = -1j*z_over_chi
                        f1 = -1j/chi
                        M = self.construct_m(f0) + self.construct_m(f1, k=1)

                ret[i*N:(i+1)*N, j*N:(j+1)*N] = M

        return ret

    def mode_transform(self, z, u, n_eq=1):
        #u[3,:] = u[3,:]/z

        return u

class DirectSolver(EigenValueSolver):
    def matrixM(self, sigma=0,
                kx=1, equilibrium=None, neglect_gas_viscosity=False):
        n_dust = len(equilibrium.tau)

        n_eq = 4 + 4*n_dust
        N = len(self.basis.collocation_points())

        a = equilibrium.viscous_alpha
        if neglect_gas_viscosity is True:
            a = 0
        b = equilibrium.beta(equilibrium.tau)
        D = equilibrium.diff(equilibrium.tau)
        S = 1.5

        # Equilibrium horizontal velocities at collocation points
        self.vx0, self.vy0, self.ux0, self.uy0 = equilibrium.evaluate(self.z)
        self.dvx0, self.dvy0, self.dux0, self.duy0 = \
          equilibrium.evaluate(self.z, k=1)
        self.d2vx0, self.d2vy0, self.d2ux0, self.d2uy0 = \
          equilibrium.evaluate(self.z, k=2)

        # rhog'/rhog
        self.P = equilibrium.dlogrhogdz(self.z)
        # (rhog'/rhog)'
        self.dP = equilibrium.dlogrhogdzdz(self.z)
        # sigma'/sigma
        self.Q = equilibrium.dlogsigmadz(self.z)
        # mu'/mu
        self.R = equilibrium.dlogmudz(self.z)

        # dust to gas ratio
        self.mu = equilibrium.epsilon(self.z)

        # Full matrix
        ret = np.zeros((N*n_eq, N*n_eq), dtype=np.cdouble)

        indptr = [0]
        indices = []
        data = []

        invA = np.eye(N)
        if self.sparse_flag == True:
            invA = np.linalg.inv(self.A)

        # Loop through all equations
        for i in range(0, n_eq):
            indptr.append(0)

            # Loop through all perturbed quantities
            for j in range(0, n_eq):
                # By default, add zeros
                M = np.zeros((N, N), dtype=np.cdouble)

                # Gas continuity equation
                if i == 0:
                    # Gas density perturbation
                    if j == 0:
                        indptr[-1] += 1
                        indices.append(j)

                        f = kx*self.vx0
                        M = self.construct_m(f)
                        data.append(invA @ M - sigma*np.eye(N))
                    # Gas mx perturbation
                    if j == 1:
                        indptr[-1] += 1
                        indices.append(j)

                        M = kx*self.A
                        data.append(invA @ M)
                    # Gas mz perturbation
                    if j == 3:
                        indptr[-1] += 1
                        indices.append(j)

                        M = -1j*self.dA
                        data.append(invA @ M)

                # Gas momentum x
                if i == 1:
                    # Gas density perturbation
                    if j == 0:
                        indptr[-1] += 1
                        indices.append(j)

                        f0 = -1j*a*(self.P*self.dvx0 + self.d2vx0) + kx
                        for n in range(0, n_dust):
                            dv = self.ux0[n,:]- self.vx0
                            f0 -= 1j*self.mu[n,:]*dv*equilibrium.weights[n]
                        M = self.construct_m(f0)
                        data.append(invA @ M)
                    # Gas mx perturbation
                    if j == 1:
                        indptr[-1] += 1
                        indices.append(j)

                        f0 = kx*self.vx0 - 4j*a*kx*kx/3
                        for n in range(0, n_dust):
                            f0 -= 1j*self.mu[n,:]*equilibrium.weights[n]
                        f1 = 1j*a*self.P
                        M = self.construct_m(f0) + \
                          self.construct_m(f1, k=1) + 1j*a*self.ddA
                        data.append(invA @ M - sigma*np.eye(N))
                    # Gas my perturbation
                    if j == 2:
                        indptr[-1] += 1
                        indices.append(j)

                        M = 2j*self.A
                        data.append(invA @ M)
                    # Gas mz perturbation
                    if j == 3:
                        indptr[-1] += 1
                        indices.append(j)

                        f0 = -1j*self.dvx0 - a*kx*self.P
                        f1 = -a*kx/3
                        M = self.construct_m(f0) - a*kx*self.dA/3
                        data.append(invA @ M)
                    # Dust density perturbation
                    if j > 3 and ((j - 4) % 4) == 0:
                        indptr[-1] += 1
                        indices.append(j)

                        n = int((j - 4)/4)
                        #f = 1j*(self.ux0[n,:] - self.vx0)/equilibrium.tau[n]
                        f = 1j*(self.ux0[n,:] - self.vx0)*equilibrium.weights[n]
                        M = self.construct_m(f)
                        data.append(invA @ M)
                    # Dust vx perturbation
                    if j > 4 and ((j - 5) % 4) == 0:   # 5, 9, 13, ...
                        indptr[-1] += 1
                        indices.append(j)

                        n = int((j - 5)/4)
                        #M = 1j*self.A/equilibrium.tau[n]
                        M = 1j*self.A*equilibrium.weights[n]
                        data.append(invA @ M)

                # Gas momentum y
                if i == 2:
                    # Gas density perturbation
                    if j == 0:
                        indptr[-1] += 1
                        indices.append(j)

                        f = -1j*a*(self.P*self.dvy0 + self.d2vy0)
                        for n in range(0, n_dust):
                            dv = self.uy0[n,:] - self.vy0
                            f -= 1j*self.mu[n,:]*dv*equilibrium.weights[n]
                        M = self.construct_m(f)
                        data.append(invA @ M)
                    # Gas vx perturbation
                    if j == 1:
                        indptr[-1] += 1
                        indices.append(j)

                        M = 1j*(S - 2)*self.A
                        data.append(invA @ M)
                    # Gas vy perturbation
                    if j == 2:
                        indptr[-1] += 1
                        indices.append(j)

                        f0 = kx*self.vx0 - 1j*a*kx*kx
                        for n in range(0, n_dust):
                            f0 -= 1j*self.mu[n,:]*equilibrium.weights[n]
                        f1 = 1j*a*self.P
                        M = self.construct_m(f0) + \
                          self.construct_m(f1, k=1) + 1j*a*self.ddA
                        data.append(invA @ M - sigma*np.eye(N))
                    # Gas vz perturbation
                    if j == 3:
                        indptr[-1] += 1
                        indices.append(j)

                        f = -1j*self.dvy0
                        M = self.construct_m(f)
                        data.append(invA @ M)
                    # Dust density perturbation
                    if j > 3 and ((j - 4) % 4) == 0:
                        indptr[-1] += 1
                        indices.append(j)

                        n = int((j - 4)/4)
                        #f = 1j*(self.uy0[n,:] - self.vy0)/equilibrium.tau[n]
                        f = 1j*(self.uy0[n,:] - self.vy0)*equilibrium.weights[n]
                        M = self.construct_m(f)
                        data.append(invA @ M)
                    # Dust vy perturbation
                    if j > 5 and ((j - 6) % 4) == 0:
                        indptr[-1] += 1
                        indices.append(j)

                        n = int((j - 6)/4)
                        #M = 1j*self.A/equilibrium.tau[n]
                        M = 1j*self.A*equilibrium.weights[n]
                        data.append(invA @ M)

                # Gas momentum z
                if i == 3:
                    # Gas density perturbation
                    if j == 0:
                        indptr[-1] += 1
                        indices.append(j)

                        f0 = 1j*self.P
                        for n in range(0, n_dust):
                            w = equilibrium.weights[n]
                            f0 += 1j*self.mu[n,:]*b[n]*self.z*w
                        M = self.construct_m(f0) - 1j*self.dA
                        data.append(invA @ M)
                    # Gas vx perturbation
                    if j == 1:
                        indptr[-1] += 1
                        indices.append(j)

                        f0 = 2*a*kx*self.P/3
                        M = self.construct_m(f0) -a*kx*self.dA/3
                        data.append(invA @ M)
                    # Gas vz perturbation
                    if j == 3:
                        indptr[-1] += 1
                        indices.append(j)

                        f0 = kx*self.vx0 - 1j*a*kx*kx
                        for n in range(0, n_dust):
                            f0 -= 1j*self.mu[n,:]*equilibrium.weights[n]
                        f1 = 4j*a*self.P/3
                        M = self.construct_m(f0) + \
                          self.construct_m(f1, k=1) + 4j*a*self.ddA/3
                        data.append(invA @ M - sigma*np.eye(N))
                    # Dust density perturbation
                    if j > 3 and ((j - 4) % 4) == 0:
                        indptr[-1] += 1
                        indices.append(j)

                        n = int((j - 4)/4)
                        #f = -1j*b[n]*self.z/equilibrium.tau[n]
                        f = -1j*b[n]*self.z*equilibrium.weights[n]
                        M = self.construct_m(f)
                        data.append(invA @ M)
                    # Dust uz perturbation
                    if j > 6 and ((j - 7) % 4) == 0:
                        indptr[-1] += 1
                        indices.append(j)

                        n = int((j - 7)/4)
                        #M = 1j*self.A/equilibrium.tau[n]
                        M = 1j*self.A*equilibrium.weights[n]
                        data.append(invA @ M)

                # Dust continuity
                if i > 3 and ((i - 4) % 4) == 0:
                    n = int((i - 4)/4)

                    # Gas density perturbation
                    if j == 0:
                        indptr[-1] += 1
                        indices.append(j)

                        f0 = 1j*D[n]*self.mu[n,:]*(kx*kx + \
                                                   self.R[n,:]*self.P + \
                                                   self.dP)
                        f1 = -1j*D[n]*self.mu[n,:]*(self.R[n,:] - self.P)
                        f2 = -1j*D[n]*self.mu[n,:]
                        M = self.construct_m(f0) + \
                          self.construct_m(f1, k=1) + self.construct_m(f2, k=2)
                        data.append(invA @ M)
                    # Dust density perturbation
                    if j == i:
                        indptr[-1] += 1
                        indices.append(j)

                        f0 = kx*self.ux0[n,:] + 1j*b[n] - \
                          1j*D[n]*(kx*kx + self.dP)
                        f1 = -1j*D[n]*self.P + 1j*b[n]*self.z
                        M = self.construct_m(f0) + self.construct_m(f1, k=1) + \
                          1j*D[n]*self.ddA
                        data.append(invA @ M - sigma*np.eye(N))
                    # Dust ux perturbation
                    if j == i + 1:
                        indptr[-1] += 1
                        indices.append(j)

                        M = kx*self.A
                        data.append(invA @ M)
                    # Dust uz perturbation
                    if j == i + 3:
                        indptr[-1] += 1
                        indices.append(j)

                        M = -1j*self.dA
                        data.append(invA @ M)

                # Dust momentum x
                if i > 4 and ((i - 5) % 4) == 0:
                    n = int((i - 5)/4)

                    # Gas vx perturbation
                    if j == 1:
                        indptr[-1] += 1
                        indices.append(j)

                        f = 1j*self.mu[n,:]/equilibrium.tau[n]
                        M = self.construct_m(f)
                        data.append(invA @ M)
                    # Dust ux perturbation
                    if j == i:
                        indptr[-1] += 1
                        indices.append(j)

                        f0 = kx*self.ux0[n,:] - 1j/equilibrium.tau[n] \
                          -1j*b[n]*self.z*self.Q[n,:]
                        f1 = 1j*b[n]*self.z
                        M = self.construct_m(f0) + self.construct_m(f1, k=1)
                        data.append(invA @ M - sigma*np.eye(N))
                    # Dust uy perturbation
                    if j == i + 1:
                        indptr[-1] += 1
                        indices.append(j)

                        M = 2j*self.A
                        data.append(invA @ M)
                    # Dust uz perturbation
                    if j == i + 2:
                        indptr[-1] += 1
                        indices.append(j)

                        f = -1j*self.dux0[n,:]
                        M = self.construct_m(f)
                        data.append(invA @ M)

                # Dust momentum y
                if i > 5 and ((i - 6) % 4) == 0:
                    n = int((i - 6)/4)

                    # Gas vy perturbation
                    if j == 2:
                        indptr[-1] += 1
                        indices.append(j)

                        f = 1j*self.mu[n,:]/equilibrium.tau[n]
                        M = self.construct_m(f)
                        data.append(invA @ M)
                    # Dust ux perturbation
                    if j == i - 1:
                        indptr[-1] += 1
                        indices.append(j)

                        M = 1j*(S - 2)*self.A
                        data.append(invA @ M)
                    # Dust uy perturbation
                    if j == i:
                        indptr[-1] += 1
                        indices.append(j)

                        f0 = kx*self.ux0[n,:] - \
                          1j/equilibrium.tau[n] - \
                          1j*b[n]*self.z*self.Q[n,:]
                        f1 = 1j*b[n]*self.z
                        M = self.construct_m(f0) + self.construct_m(f1, k=1)
                        data.append(invA @ M - sigma*np.eye(N))
                    # Dust uz perturbation
                    if j == i + 1:
                        indptr[-1] += 1
                        indices.append(j)

                        f = -1j*self.duy0[n,:]
                        M = self.construct_m(f)
                        data.append(invA @ M)

                # Dust momentum z
                if i > 6 and ((i - 7) % 4) == 0:
                    n = int((i - 7)/4)

                    # Gas vz perturbation
                    if j == 3:
                        indptr[-1] += 1
                        indices.append(j)

                        f = 1j*self.mu[n,:]/equilibrium.tau[n]
                        M = self.construct_m(f)
                        data.append(invA @ M)
                    # Dust uz perturbation
                    if j == i:
                        indptr[-1] += 1
                        indices.append(j)

                        f0 = kx*self.ux0[n,:] - \
                          1j/equilibrium.tau[n] + 1j*b[n] - \
                          1j*b[n]*self.z*self.Q[n,:]
                        f1 = 1j*b[n]*self.z
                        M = self.construct_m(f0) + self.construct_m(f1, k=1)
                        data.append(invA @ M - sigma*np.eye(N))

                ret[i*N:(i+1)*N, j*N:(j+1)*N] = M

        data = np.asarray(data)
        indices = np.asarray(indices)
        indptr = np.cumsum(np.asarray(indptr))

        ret = sparse.bsr_matrix((data, indices, indptr), shape=(n_eq*N, n_eq*N))

        if self.sparse_flag == False:
            ret = ret.todense()

        return ret
