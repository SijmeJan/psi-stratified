import numpy as np
import h5py as h5

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
