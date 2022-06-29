import numpy as np
import h5py as h5

def unique_within_tol_complex(a, tol=1e-12):
    """Brute force finding unique complex values"""
    if a.size == 0:
        return a

    sel = np.ones(len(a), dtype=bool)

    for i in range(0, len(a)-1):
        dist = np.abs(a[i+1:] - a[i]).tolist()
        if np.min(dist) < tol:
            sel[i] = False

    return a[sel], sel

class ModeFinder():
    def __init__(self, strat_box):
        self.sb = strat_box

    def find_optimal_L(self, wave_number_x, N, nL=10):
        L = np.logspace(-3, 0, nL)
        n_found = 0*L

        for i in range(0, nL):
            self.sb.find_eigenvalues(wave_number_x=wave_number_x,
                                     N=N,
                                     L=L[i],
                                     n_dust=self.sb.n_dust,
                                     sparse_flag=True,
                                     n_safe_levels=2,
                                     use_PETSc=True,
                                     sigma=0 + 0.001j,
                                     n_eig=-1)
            n_found[i] = len(self.sb.eig)
            print(i, L[i], n_found[i])

        return L[np.argmax(n_found)]

    def find_all(self, wave_number):
        kx = wave_number

        N = 50
        L = 1.0

        # Go through range of L to find best value

    def find_growing_at_real_part(self, real_part,
                                  wave_number_x, N, n_eig=10):
        max_n_eig = 4*(self.sb.n_dust + 1)*N

        max_growth = 2.0
        imag_part = 0.0

        e_val = []
        e_vec = []

        while imag_part < max_growth:
            sigma = real_part + 1j*imag_part
            self.sb.find_eigenvalues(wave_number_x=wave_number_x,
                                     N=N,
                                     L=self.sb.L,
                                     n_dust=self.sb.n_dust,
                                     sparse_flag=True,
                                     n_safe_levels=1,
                                     use_PETSc=True,
                                     sigma=sigma,
                                     n_eig=n_eig)

            while len(self.sb.eig) == 0 and n_eig < max_n_eig:
                n_eig = n_eig*2
                if n_eig > max_n_eig:
                    n_eig=max_n_eig

                self.sb.find_eigenvalues(wave_number_x=wave_number_x,
                                         N=N,
                                         L=self.sb.L,
                                         n_dust=self.sb.n_dust,
                                         sparse_flag=True,
                                         n_safe_levels=1,
                                         use_PETSc=True,
                                         sigma=sigma,
                                         n_eig=n_eig)


            if len(self.sb.eig) == 0:
                print('No eigenvalues found!')
                return [], []

            print(imag_part, n_eig)

            e_val.extend(self.sb.eig.tolist())
            e_vec.extend(self.sb.vec.tolist())

            rad = np.max(np.abs(self.sb.eig - sigma))
            imag_part = imag_part + rad

        # Prune to unique eigenvalues
        e_val, sel = unique_within_tol_complex(np.asarray(e_val))
        e_vec = np.asarray(e_vec)[sel,:]

        return e_val, e_vec
