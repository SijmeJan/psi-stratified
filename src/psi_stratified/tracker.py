import numpy as np

from scipy.interpolate import interp1d

class ModeTracker():
    def __init__(self, strat_box, filename=None):
        self.sb = strat_box
        self.filename = filename

    def safe_step(self, sigma, maxN=600):
        self.sb.find_eigenvalues(wave_number_x=self.sb.kx,
                                 N=self.sb.N,
                                 L=self.sb.L,
                                 n_dust=self.sb.n_dust,
                                 sparse_flag=True,
                                 use_PETSc=True,
                                 sigma=sigma,
                                 n_eig=1)

        original_L = self.sb.L

        if len(np.atleast_1d(self.sb.eig)) == 0:
            self.sb.find_eigenvalues(wave_number_x=self.sb.kx,
                                     N=self.sb.N,
                                     L=self.sb.L/1.5,
                                     n_dust=self.sb.n_dust,
                                     sparse_flag=True,
                                     use_PETSc=True,
                                     sigma=sigma,
                                     n_eig=1)

            if len(np.atleast_1d(self.sb.eig)) == 0:
                # Lowering L did not work; try increasing L
                self.sb.find_eigenvalues(wave_number_x=self.sb.kx,
                                         N=self.sb.N,
                                         L=self.sb.L*1.5*1.5,
                                         n_dust=self.sb.n_dust,
                                         sparse_flag=True,
                                         use_PETSc=True,
                                         sigma=sigma,
                                         n_eig=1)

            if len(np.atleast_1d(self.sb.eig)) == 0:
                # Try higher resolution
                higher_N = self.sb.N + 50
                while (len(np.atleast_1d(self.sb.eig))==0 and higher_N <= maxN):
                    self.sb.find_eigenvalues(wave_number_x=self.sb.kx,
                                             N=higher_N,
                                             L=original_L,
                                             n_dust=self.sb.n_dust,
                                             sparse_flag=True,
                                             use_PETSc=True,
                                             sigma=sigma,
                                             n_eig=1)
                    higher_N = self.sb.N + 50

                if len(np.atleast_1d(self.sb.eig)) == 0:
                    print('Forcing closest eigenvalue at highest res')
                    self.sb.eig = self.sb.di.eval_hires
        else:
            if self.sb.N > 50:
                self.sb.N = self.sb.N - 50

        # Select closest to sigma
        e = np.atleast_1d(self.sb.eig)
        k = np.argmin(np.abs(e - sigma))

        return self.sb.eig[k]

    def track(self):
        pass


class WaveNumberTracker(ModeTracker):
    def track(self, wave_numbers, starting_ev, maxN=600):
        # Wavenumbers to evaluate modes at
        kx = np.atleast_1d(wave_numbers)

        # Starting eigenvalues, corresponding to wave_numbers[0]
        ev = np.atleast_1d(starting_ev)

        ret = np.zeros((len(ev), len(kx)), dtype=np.cdouble)
        ret[:,0] = ev

        for j in range(0, len(ev)):
            for i in range(1, len(kx)):
                self.sb.kx = kx[i]

                # Guess where next ev could be
                target = ret[j, i-1]
                if i > 1:
                    target = \
                      interp1d(kx[0:i], ret[j, 0:i], \
                               fill_value='extrapolate')(kx[i])
                    #target = ret[j, i-1] + \
                    #  (ret[j, i-1] - ret[j, i-2])/(kx[i-1] - kx[i-2])*(kx[i] - kx[i-1])

                ret[j, i] = self.safe_step(target, maxN=maxN)

                # Save to file.
                # Not sure how to do it with other parameters than wavenumber?
                if self.filename is not None:
                    self.sb.save(self.filename)

                print(i, self.sb.N, self.sb.L, kx[i], ret[j, i], flush=True)

        return ret

class StokesRangeTracker(ModeTracker):
    def track(self, stokes_min, stokes_max, starting_ev, maxN=600):
        #print('Starting tracker...')

        st_min = np.atleast_1d(stokes_min)
        st_max = np.atleast_1d(stokes_max)

        # Starting eigenvalues, corresponding to wave_numbers[0]
        ev = np.atleast_1d(starting_ev)

        ret = np.zeros((len(ev), len(st_max)), dtype=np.cdouble)
        ret[:,0] = ev

        for j in range(0, len(ev)):
            for i in range(1, len(st_max)):
                #print('Setting Stokes Range...')
                self.sb.set_stokes_range([st_min[i], st_max[i]])

                #print('Performing safe step...')
                ret[j, i] = self.safe_step(ret[j, i-1], maxN=maxN)


                print(i, self.sb.N, st_min[i], st_max[i], ret[j,i])
                #print(self.sb.eig, self.sb.vec)
        return ret

class DustFluidTracker(ModeTracker):
    def track(self, n_dust, starting_ev, maxN=600):
        n_dust = np.atleast_1d(n_dust)

        # Starting eigenvalues, corresponding to wave_numbers[0]
        ev = np.atleast_1d(starting_ev)

        ret = np.zeros((len(ev), len(n_dust)), dtype=np.cdouble)
        ret[:,0] = ev

        for j in range(0, len(ev)):
            for i in range(1, len(n_dust)):
                self.sb.set_n_dust(n_dust[i])

                ret[j, i] = self.safe_step(ret[j, i-1], maxN=maxN)

                print(i, self.sb.N, n_dust[i], ret[j,i])
        return ret

class ViscosityTracker(ModeTracker):
    def track(self, viscous_alpha, starting_ev, maxN=600):
        viscous_alpha = np.atleast_1d(viscous_alpha)

        # Starting eigenvalues, corresponding to wave_numbers[0]
        ev = np.atleast_1d(starting_ev)

        ret = np.zeros((len(ev), len(viscous_alpha)), dtype=np.cdouble)
        ret[:,0] = ev

        for j in range(0, len(ev)):
            for i in range(1, len(viscous_alpha)):
                self.sb.set_viscosity(viscous_alpha[i])
                self.sb.L = 0.006*np.sqrt(viscous_alpha[i]/1.0e-6)

                ret[j, i] = self.safe_step(ret[j, i-1], maxN=maxN)

                print(i, self.sb.N, viscous_alpha[i], ret[j,i])
        return ret
