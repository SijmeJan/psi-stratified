import numpy as np
import h5py as h5

from scipy.interpolate import interp1d

class TrackerFile():
    def __init__(self, filename):
        self.filename = filename

    def save(self, x_values, y_values, group_name):
        hf = h5.File(self.filename, 'a')

        if group_name not in hf:
            g = hf.create_group(group_name)
        else:
            g = hf[group_name]
            print('Warning: replacing eigenvalues and eigenvectors!')
            del g['x_values']
            del g['y_values']

        g.create_dataset('x_values', data=x_values)
        g.create_dataset('y_values', data=y_values)

        hf.close()

    def save_single(self, x_value, y_value, group_name):
        '''Insert single (x,y) pair in existing sequence'''
        hf = h5.File(self.filename, 'a')

        if group_name not in hf:
            #print('tf: Creating new group')
            g = hf.create_group(group_name)

            x_values = np.atleast_1d(np.asarray(x_value))
            y_values = np.atleast_1d(np.asarray(y_value))
        else:
            #print('tf: Adding to existing group')
            g = hf[group_name]

            x_values = g.get('x_values')[()]
            y_values = g.get('y_values')[()]

            if x_value not in np.atleast_1d(x_values):
                #print('tf: inserting')
                idx = np.searchsorted(x_values, x_value)

                x_values = np.insert(x_values, idx, x_value)
                y_values = np.insert(y_values, idx, y_value)

            del g['x_values']
            del g['y_values']

        g.create_dataset('x_values', data=x_values)
        g.create_dataset('y_values', data=y_values)

        hf.close()

    def get_single(self, x_value, group_name):
        hf = h5.File(self.filename, 'a')

        ret = None
        if group_name in hf:
            x_in_file = hf[group_name].get('x_values')[()]
            if x_value in np.atleast_1d(x_in_file):
                y_in_file = hf[group_name].get('y_values')[()]

                ret = y_in_file[np.asarray(x_in_file == x_value).nonzero()[0]]

        hf.close()

        return ret

    def delete_group(self, group_name):
        hf = h5.File(self.filename, 'a')

        if group_name in hf:
            g = hf[group_name]
            del g['x_values']
            del g['y_values']
        else:
            print(group_name, ' not found in file')

        hf.close()

    def merge_groups(self, group_list, group_merged):
        hf = h5.File(self.filename, 'a')

        x_merged = []
        y_merged = []
        for group in group_list:
            x_merged.extend(hf[group].get('x_values')[()].tolist())
            y_merged.extend(hf[group].get('y_values')[()].tolist())

        hf.close()

        #for group in group_list:
        #    self.delete_group(group)

        self.save(x_merged, y_merged, group_merged)

    def list_groups(self):
        hf = h5.File(self.filename, 'r')

        for g in hf:
            print(g)

class ModeTracker():
    def __init__(self, strat_box, filename=None):
        self.sb = strat_box
        #self.filename = filename

        self.tf = None
        if filename is not None:
            self.tf = TrackerFile(filename)

    def safe_step(self, sigma, maxN=600):
        #print('mt: finding eigenvalues at N = ', self.sb.N)
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
            #print('mt: finding eigenvalues at L = ', self.sb.L/1.5)
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
                #print('mt: finding eigenvalues at L = ', self.sb.L*1.5*1.5)
                self.sb.find_eigenvalues(wave_number_x=self.sb.kx,
                                         N=self.sb.N,
                                         L=self.sb.L*1.5*1.5,
                                         n_dust=self.sb.n_dust,
                                         sparse_flag=True,
                                         use_PETSc=True,
                                         sigma=sigma,
                                         n_eig=1)

            if len(np.atleast_1d(self.sb.eig)) == 0:
                # Varying L did not help, restore original
                self.sb.L = original_L

                # Try higher resolution
                higher_N = self.sb.N + 50
                while (len(np.atleast_1d(self.sb.eig))==0 and higher_N <= maxN):
                    #print('mt: finding eigenvalues at N = ', higher_N)
                    self.sb.find_eigenvalues(wave_number_x=self.sb.kx,
                                             N=higher_N,
                                             L=self.sb.L,
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

    def save(self, x_values, y_values, group_name):
        if self.filename is not None:
            hf = h5.File(self.filename, 'a')

            if group_name not in hf:
                g = hf.create_group(group_name)
            else:
                g = hf[group_name]
                print('Warning: replacing eigenvalues and eigenvectors!')
                del g['x_values']
                del g['y_values']

            g.create_dataset('x_values', data=x_values)
            g.create_dataset('y_values', data=y_values)

            hf.close()

class WaveNumberTracker(ModeTracker):
    def track(self, wave_numbers, starting_ev, maxN=600, label='main'):
        # Wavenumbers to evaluate modes at
        kx = np.atleast_1d(wave_numbers)

        # Starting eigenvalues, corresponding to wave_numbers[0]
        ev = np.atleast_1d(starting_ev)

        ret = np.zeros((len(ev), len(kx)), dtype=np.cdouble)
        ret[:,0] = ev


        for j in range(0, len(ev)):
            self.tf.save_single(kx[0], ret[j,0], label)
            for i in range(1, len(kx)):
                self.sb.kx = kx[i]

                y = self.tf.get_single(kx[i], label)

                if y == None:
                    # Guess where next ev could be
                    target = ret[j, i-1]
                    if i > 1:
                        if kx[i-1] != kx[i]:
                            target = \
                              interp1d(kx[0:i], ret[j, 0:i], \
                                       fill_value='extrapolate')(kx[i])

                    ret[j, i] = self.safe_step(target, maxN=maxN)


                    self.tf.save_single(kx[i], ret[j,i], label)
                else:
                    ret[j,i] = y

                print(i, self.sb.N, self.sb.L, kx[i], ret[j, i], flush=True)

        #self.save(kx, ret, label)

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
