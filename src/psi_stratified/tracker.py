import numpy as np
import h5py as h5

from scipy.interpolate import interp1d

class TrackerFile():
    def __init__(self, filename):
        self.filename = filename

    def save(self, x_values, y_values, N, L, group_name):
        hf = h5.File(self.filename, 'a')

        if group_name not in hf:
            g = hf.create_group(group_name)
        else:
            g = hf[group_name]
            print('Warning: replacing eigenvalues and eigenvectors!')
            del g['x_values']
            del g['y_values']
            del g['N']
            del g['L']


        g.create_dataset('x_values', data=x_values)
        g.create_dataset('y_values', data=y_values)
        g.create_dataset('N', data=N)
        g.create_dataset('L', data=L)

        hf.close()

    def save_single(self, x_value, y_value, N, L, group_name):
        '''Insert single (x,y) pair in existing sequence'''
        hf = h5.File(self.filename, 'a')

        if group_name not in hf:
            #print('tf: Creating new group')
            g = hf.create_group(group_name)

            x_values = np.atleast_1d(np.asarray(x_value))
            y_values = np.atleast_1d(np.asarray(y_value))
            N_values = np.atleast_1d(np.asarray(N))
            L_values = np.atleast_1d(np.asarray(L))
        else:
            #print('tf: Adding to existing group')
            g = hf[group_name]

            x_values = g.get('x_values')[()]
            y_values = g.get('y_values')[()]
            N_values = g.get('N')[()]
            L_values = g.get('L')[()]

            if x_value not in np.atleast_1d(x_values):
                #print('tf: inserting')
                idx = np.searchsorted(x_values, x_value)

                x_values = np.insert(x_values, idx, x_value)
                y_values = np.insert(y_values, idx, y_value)
                N_values = np.insert(N_values, idx, N)
                L_values = np.insert(L_values, idx, L)

            del g['x_values']
            del g['y_values']
            del g['N']
            del g['L']

        g.create_dataset('x_values', data=x_values)
        g.create_dataset('y_values', data=y_values)
        g.create_dataset('N', data=N_values)
        g.create_dataset('L', data=L_values)

        hf.close()

    def get_single(self, x_value, group_name):
        hf = h5.File(self.filename, 'a')

        ret = None
        if group_name in hf:
            x_in_file = hf[group_name].get('x_values')[()]
            if x_value in np.atleast_1d(x_in_file):
                idx = np.asarray(x_in_file == x_value).nonzero()[0]

                y_in_file = hf[group_name].get('y_values')[()]
                N_in_file = hf[group_name].get('N')[()]
                L_in_file = hf[group_name].get('L')[()]

                ret = y_in_file[idx], N_in_file[idx], L_in_file[idx]

        hf.close()

        return ret

    def delete_group(self, group_name):
        hf = h5.File(self.filename, 'a')

        if group_name in hf:
            g = hf[group_name]
            del g['x_values']
            del g['y_values']
            del g['N']
            del g['L']
            del hf[group_name]
        else:
            print(group_name, ' not found in file')

        hf.close()

    def merge_groups(self, group_list, group_merged):
        hf = h5.File(self.filename, 'a')

        x_merged = []
        y_merged = []
        N_merged = []
        L_merged = []
        for group in group_list:
            x_merged.extend(hf[group].get('x_values')[()].tolist())
            y_merged.extend(hf[group].get('y_values')[()].tolist())
            N_merged.extend(hf[group].get('N')[()].tolist())
            L_merged.extend(hf[group].get('L')[()].tolist())

        hf.close()

        self.save(x_merged, y_merged, group_merged)

    def list_groups(self):
        hf = h5.File(self.filename, 'r')

        groups = []
        for g in hf:
            groups.append(g)


        hf.close()

        return groups

    def rename_group(self, group_old, group_new):
        hf = h5.File(self.filename, 'a')

        if group_old in hf:
            x, y, N, L = self.read_group(group_old)
        else:
            print('Could not rename group ', group_old)
            hf.close()
            return

        self.save(x, y, N, L, group_new)
        self.delete_group(group_old)

    def read_group(self, group_name):
        hf = h5.File(self.filename, 'r')

        if group_name in hf:
            g = hf[group_name]

            # Get data
            x = g.get('x_values')[()]
            y = g.get('y_values')[()]
            N = g.get('N')[()]
            L = g.get('L')[()]

            hf.close()
            return x, y, N, L
        else:
            print('Could not read group ', group_name)
            hf.close()
            return None

    def add_from_file(self, tf_add):
        for group_add in tf_add.list_groups():
            x, y, N, L = tf_add.read_group(group_add)
            for i in range(0,len(x)):
                self.save_single(x[i], y[i], N[i], L[i], group_add)

class ModeTracker():
    def __init__(self, strat_box, filename=None):
        self.sb = strat_box

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

            used_N = self.sb.N
            used_L = self.sb.L
        else:
            used_N = self.sb.N
            used_L = self.sb.L

            if self.sb.N > 50:
                self.sb.N = self.sb.N - 50

        # Select closest to sigma
        e = np.atleast_1d(self.sb.eig)
        k = np.argmin(np.abs(e - sigma))

        return self.sb.eig[k], used_N, used_L

    def adjust_sbox(self, x_value):
        pass

    def get_save_xvalue(self, x_value):
        return x_value

    def track(self, x_values, starting_ev, maxN=600, label='main'):
        # Wavenumbers to evaluate modes at
        xv = np.atleast_1d(x_values)

        # Starting eigenvalues, corresponding to wave_numbers[0]
        ev = np.atleast_1d(starting_ev)

        ret = np.zeros((len(ev), len(xv)), dtype=np.cdouble)
        ret[:,0] = ev

        for j in range(0, len(ev)):
            if self.tf is not None:
                save_xv = self.get_save_xvalue(xv[0])
                self.tf.save_single(save_xv, ret[j,0], self.sb.N,
                                    self.sb.L, label)
            for i in range(1, len(xv)):
                self.adjust_sbox(xv[i])

                y = None
                if self.tf is not None:
                    y = self.tf.get_single(self.get_save_xvalue(xv[i]), label)

                if y == None:
                    # Guess where next ev could be
                    target = ret[j, i-1]
                    if i > 1:
                        if xv[i-1] != xv[i]:
                            target = \
                              interp1d(xv[0:i], ret[j, 0:i], \
                                       fill_value='extrapolate')(xv[i])

                    ret[j, i], N, L = self.safe_step(target, maxN=maxN)


                    if self.tf is not None:
                        save_xv = self.get_save_xvalue(xv[i])
                        self.tf.save_single(save_xv, ret[j,i], N, L, label)
                else:
                    ret[j, i], self.sb.N, self.sb.L = y

                print(i, self.sb.N, self.sb.L, xv[i], ret[j, i], flush=True)

        return ret

class WaveNumberTracker(ModeTracker):
    def adjust_sbox(self, x_value):
        self.sb.kx = x_value

class StokesRangeTracker(ModeTracker):
    def adjust_sbox(self, x_value):
        # Vary st_min, keep st_max fixed
        st_max = self.sb.param['stokes_range'][1]
        self.sb.set_stokes_range([x_value, st_max])

    def get_save_xvalue(self, x_value):
        x_value = 1000*np.log10(x_value)
        return int(x_value)

class DustFluidTracker(ModeTracker):
    def adjust_sbox(self, x_value):
        self.sb.set_n_dust(x_value)

class ViscosityTracker(ModeTracker):
    def adjust_sbox(self, x_value):
        self.sb.set_viscosity(x_value)

    def get_save_xvalue(self, x_value):
        x_value = 1000*np.log10(x_value)
        return int(x_value)
