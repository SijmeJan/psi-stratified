# -*- coding: utf-8 -*-
"""Tracking eigenvalues when parameters change.
"""
import numpy as np
import h5py as h5

from scipy.interpolate import interp1d

class TrackerFile():
    """Class dealing with eigenvalues computed using tracker

    Args:
        filename: name of file
    """
    def __init__(self, filename):
        self.filename = filename

    def save(self, x_values, y_values, n_coll, scale_l, group_name):
        """Save series of results into file

        If group_name exists in file, it will be replaced.

        Args:
            x_values: values of parameter that is changing
            y_values: corresponding eigenvalue
            n_coll: number of collocation points used
            scale_l: scale factor used
            group_name: name of group in HDF5 file

        Returns:
            nothing
        """
        h5_file = h5.File(self.filename, 'a')

        # Create group, or replace if exists
        if group_name not in h5_file:
            grp = h5_file.create_group(group_name)
        else:
            grp = h5_file[group_name]
            print('Warning: replacing eigenvalues and eigenvectors!')
            del grp['x_values']
            del grp['y_values']
            del grp['n_coll']
            del grp['scale_l']


        grp.create_dataset('x_values', data=x_values)
        grp.create_dataset('y_values', data=y_values)
        grp.create_dataset('n_coll', data=n_coll)
        grp.create_dataset('scale_l', data=scale_l)

        h5_file.close()

    def save_single(self, x_value, y_value, n_coll, scale_l, group_name):
        """Insert single (x,y) pair in existing sequence

        Args:
            x_value: value of parameter that is changing
            y_value: corresponding eigenvalue
            n_coll: number of collocation points used
            scale_l: scale factor used
            group_name: name of group in HDF5 file

        """
        h5_file = h5.File(self.filename, 'a')

        if group_name not in h5_file:
            grp = h5_file.create_group(group_name)

            x_values = np.atleast_1d(np.asarray(x_value))
            y_values = np.atleast_1d(np.asarray(y_value))
            n_coll_values = np.atleast_1d(np.asarray(n_coll))
            scale_l_values = np.atleast_1d(np.asarray(scale_l))
        else:
            #print('t_file: Adding to existing group')
            grp = h5_file[group_name]

            x_values = grp.get('x_values')[()]
            y_values = grp.get('y_values')[()]
            n_coll_values = grp.get('n_coll')[()]
            scale_l_values = grp.get('scale_l')[()]

            if x_value not in np.atleast_1d(x_values):
                #print('t_file: inserting')
                idx = np.searchsorted(x_values, x_value)

                x_values = np.insert(x_values, idx, x_value)
                y_values = np.insert(y_values, idx, y_value)
                n_coll_values = np.insert(n_coll_values, idx, n_coll)
                scale_l_values = np.insert(scale_l_values, idx, scale_l)

            del grp['x_values']
            del grp['y_values']
            del grp['n_coll']
            del grp['scale_l']

        grp.create_dataset('x_values', data=x_values)
        grp.create_dataset('y_values', data=y_values)
        grp.create_dataset('n_coll', data=n_coll_values)
        grp.create_dataset('scale_l', data=scale_l_values)

        h5_file.close()

    def get_single(self, x_value, group_name):
        """Read single entry from file

        Args:
            x_value: parameter to search at
            group_name: group to search in

        Returns:
            If x_value is found in file: eigenvalue, number of collocation
            points, scale factor. Otherwise, None.
        """
        h5_file = h5.File(self.filename, 'a')

        ret = None
        if group_name in h5_file:
            x_in_file = h5_file[group_name].get('x_values')[()]
            if x_value in np.atleast_1d(x_in_file):
                idx = np.asarray(x_in_file == x_value).nonzero()[0]

                y_in_file = h5_file[group_name].get('y_values')[()]
                n_coll_in_file = h5_file[group_name].get('n_coll')[()]
                scale_l_in_file = h5_file[group_name].get('scale_l')[()]

                ret = y_in_file[idx], n_coll_in_file[idx], scale_l_in_file[idx]

        h5_file.close()

        return ret

    def delete_group(self, group_name):
        """Delete group from HDF5 file

        Args:
            group_name: name of group to delete
        """
        h5_file = h5.File(self.filename, 'a')

        if group_name in h5_file:
            grp = h5_file[group_name]
            del grp['x_values']
            del grp['y_values']
            del grp['n_coll']
            del grp['scale_l']
            del h5_file[group_name]
        else:
            print(group_name, ' not found in file')

        h5_file.close()

    def merge_groups(self, group_list, group_merged):
        """Merge groups in HDF5 file

        Args:
            group_list: list of group names to merge
            group_merged: name of merged group
        """
        h5_file = h5.File(self.filename, 'a')

        x_merged = []
        y_merged = []
        n_coll_merged = []
        scale_l_merged = []
        for group in group_list:
            x_merged.extend(h5_file[group].get('x_values')[()].tolist())
            y_merged.extend(h5_file[group].get('y_values')[()].tolist())
            n_coll_merged.extend(h5_file[group].get('n_coll')[()].tolist())
            scale_l_merged.extend(h5_file[group].get('scale_l')[()].tolist())

        h5_file.close()

        self.save(x_merged, y_merged, group_merged)

    def list_groups(self):
        """List all groups in HDF5 file

        Returns:
            list of group names
        """
        h5_file = h5.File(self.filename, 'r')

        groups = []
        for grp in h5_file:
            groups.append(grp)

        h5_file.close()

        return groups

    def rename_group(self, group_old, group_new):
        """Rename a group in HDF5 file

        Args:
            group_old: old group name
            group_new: new group name
        """
        h5_file = h5.File(self.filename, 'a')

        if group_old in h5_file:
            x_v, y_v, n_coll, scale_l = self.read_group(group_old)
        else:
            print('Could not rename group ', group_old)
            h5_file.close()
            return

        self.save(x_v, y_v, n_coll, scale_l, group_new)
        self.delete_group(group_old)

    def read_group(self, group_name):
        """Read group data from HDF5 file

        Args:
            group_name: name of group to read from

        Returns:
            parameter values, eigenvalues, number of collocation points,
            scale factors. Returns None if group_name is not found.
        """
        h5_file = h5.File(self.filename, 'r')

        if group_name in h5_file:
            grp = h5_file[group_name]

            # Get data
            x_v = grp.get('x_values')[()]
            y_v = grp.get('y_values')[()]
            n_coll = grp.get('n_coll')[()]
            scale_l = grp.get('scale_l')[()]

            h5_file.close()
            return x_v, y_v, n_coll, scale_l

        print('Could not read group ', group_name)
        h5_file.close()
        return None

    def add_from_file(self, t_file_add):
        """Add groups from different HDF5 file

        Args:
            t_file_add: TrackerFile object of file to add
        """
        for group_add in t_file_add.list_groups():
            x_v, y_v, n_coll, scale_l = t_file_add.read_group(group_add)
            for i, scl in enumerate(scale_l):
                self.save_single(x_v[i], y_v[i], n_coll[i], scl, group_add)

class ModeTracker():
    """Base class for tracking modes across parameter space.

    Args:
        strat_box: StratBox object for calculating modes
        filename: name of HDF5 file to store results in
    """
    def __init__(self, strat_box, filename=None):
        self.sbx = strat_box

        self.t_file = None
        if filename is not None:
            self.t_file = TrackerFile(filename)

    def safe_step(self, sigma, maxn_coll=600):
        """Find eigenvalue close to sigma

        Args:
            sigma: guessed eigenvalue
            maxn_coll: maximum collocation points to consider

        Returns:
            eigenvalue, number of collocation points used, scale factor used
        """
        #print('mt: finding eigenvalues at n_coll = ', self.sbx.n_coll)
        self.sbx.find_eigenvalues(wave_number_x=self.sbx.kx,
                                 n_coll=self.sbx.n_coll,
                                 scale_l=self.sbx.scale_l,
                                 n_dust=self.sbx.n_dust,
                                 sparse_flag=True,
                                 use_PETSc=True,
                                 sigma=sigma,
                                 n_eig=1)

        original_scale_l = self.sbx.scale_l

        if len(np.atleast_1d(self.sbx.eig)) == 0:
            #print('mt: finding eigenvalues at scale_l = ', self.sbx.scale_l/1.5)
            self.sbx.find_eigenvalues(wave_number_x=self.sbx.kx,
                                     n_coll=self.sbx.n_coll,
                                     scale_l=self.sbx.scale_l/1.5,
                                     n_dust=self.sbx.n_dust,
                                     sparse_flag=True,
                                     use_PETSc=True,
                                     sigma=sigma,
                                     n_eig=1)

            if len(np.atleast_1d(self.sbx.eig)) == 0:
                # Lowering L did not work; try increasing L
                #print('mt: finding eigenvalues at scale_l = ', self.sbx.scale_l*1.5*1.5)
                self.sbx.find_eigenvalues(wave_number_x=self.sbx.kx,
                                         n_coll=self.sbx.n_coll,
                                         scale_l=self.sbx.scale_l*1.5*1.5,
                                         n_dust=self.sbx.n_dust,
                                         sparse_flag=True,
                                         use_PETSc=True,
                                         sigma=sigma,
                                         n_eig=1)

            if len(np.atleast_1d(self.sbx.eig)) == 0:
                # Varying L did not help, restore original
                self.sbx.scale_l = original_scale_l

                # Try higher resolution
                higher_n_coll = self.sbx.n_coll + 50
                while (len(np.atleast_1d(self.sbx.eig))==0 and higher_n_coll <= maxn_coll):
                    #print('mt: finding eigenvalues at n_coll = ', higher_n_coll)
                    self.sbx.find_eigenvalues(wave_number_x=self.sbx.kx,
                                             n_coll=higher_n_coll,
                                             scale_l=self.sbx.scale_l,
                                             n_dust=self.sbx.n_dust,
                                             sparse_flag=True,
                                             use_PETSc=True,
                                             sigma=sigma,
                                             n_eig=1)
                    higher_n_coll = self.sbx.n_coll + 50

                if len(np.atleast_1d(self.sbx.eig)) == 0:
                    print('Forcing closest eigenvalue at highest res')
                    self.sbx.eig = self.sbx.di.eval_hires

            used_n_coll = self.sbx.n_coll
            used_scale_l = self.sbx.scale_l
        else:
            used_n_coll = self.sbx.n_coll
            used_scale_l = self.sbx.scale_l

            if self.sbx.n_coll > 50:
                self.sbx.n_coll = self.sbx.n_coll - 50

        # Select closest to sigma
        e_val = np.atleast_1d(self.sbx.eig)
        idx_min = np.argmin(np.abs(e_val - sigma))

        return self.sbx.eig[idx_min], used_n_coll, used_scale_l

    def adjust_sbox(self, x_value):
        """Change the required parameter in the ShearingBox"""

    def get_save_xvalue(self, x_value):
        """Compute parameter value to store in file

        We need to be able to identify the parameters in the HDF file exactly.
        Sometimes it is necessary to truncate the floating point value to
        achieve this.
        """
        return x_value

    def track(self, x_values, starting_ev, maxn_coll=600, label='main'):
        """Track a mode across parameter space

        Args:
            x_values: parameter range at which to follow mode
            starting_ev: eigenvalues at x_values[0]
            maxn_coll: maximum number of collocation points to consider
            label: group name to create in HDF5 file (if saving to file)

        Returns:
            eigenvalues at parameter values.
        """
        # Parameters to evaluate modes at
        x_val = np.atleast_1d(x_values)

        # Starting eigenvalues, corresponding to x_val[0]
        e_val = np.atleast_1d(starting_ev)

        ret = np.zeros((len(e_val), len(x_val)), dtype=np.cdouble)
        ret[:,0] = e_val

        for j in range(0, len(e_val)):
            if self.t_file is not None:
                save_xv = self.get_save_xvalue(x_val[0])
                self.t_file.save_single(save_xv, ret[j,0], self.sbx.n_coll,
                                    self.sbx.scale_l, label)
            for i in range(1, len(x_val)):
                self.adjust_sbox(x_val[i])

                y_val = None
                if self.t_file is not None:
                    y_val = self.t_file.get_single(self.get_save_xvalue(x_val[i]), label)

                if y_val is None:
                    # Guess where next ev could be
                    target = ret[j, i-1]
                    if i > 1:
                        if x_val[i-1] != x_val[i]:
                            target = \
                              interp1d(x_val[0:i], ret[j, 0:i], \
                                       fill_value='extrapolate')(x_val[i])

                    ret[j, i], n_coll, scale_l = self.safe_step(target, maxn_coll=maxn_coll)


                    if self.t_file is not None:
                        save_xv = self.get_save_xvalue(x_val[i])
                        self.t_file.save_single(save_xv, ret[j,i], n_coll, scale_l, label)
                else:
                    ret[j, i], self.sbx.n_coll, self.sbx.scale_l = y_val

                print(i, self.sbx.n_coll, self.sbx.scale_l, x_val[i], ret[j, i], flush=True)

        return ret

class WaveNumberTracker(ModeTracker):
    """Track modes across wave number space"""
    def adjust_sbox(self, x_value):
        """Adjust the wave number in the StratBox"""
        self.sbx.kx = x_value

class StokesRangeTracker(ModeTracker):
    """Track modes across Stokes range space"""
    def adjust_sbox(self, x_value):
        """Adjust Stokes range in StratBox"""

        # Vary st_min, keep st_max fixed
        st_max = self.sbx.param['stokes_range'][1]
        self.sbx.set_stokes_range([x_value, st_max])

    def get_save_xvalue(self, x_value):
        """Save the base 10 log times 1000"""
        x_value = 1000*np.log10(x_value)
        return int(x_value)

class DustFluidTracker(ModeTracker):
    """Track modes while varying the number of dust fluids"""
    def adjust_sbox(self, x_value):
        """Set number of dust fluids in StratBox"""
        self.sbx.set_n_dust(x_value)

class ViscosityTracker(ModeTracker):
    """Track modes varying the viscosity"""
    def adjust_sbox(self, x_value):
        """Set new viscosity in StratBox"""
        self.sbx.set_viscosity(x_value)

    def get_save_xvalue(self, x_value):
        """Save the base 10 log times 1000"""
        x_value = 1000*np.log10(x_value)
        return int(x_value)
