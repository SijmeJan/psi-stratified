# -*- coding: utf-8 -*-
"""Module containing StratBox class.
"""

import numpy as np
import h5py as h5

from .equilibrium import Equilibrium
from .direct import DirectSolver
from .stokesdensity import StokesDensity

class StratBox():
    """Shearing box linear modes for the stratified PSI.

    A StratBox instance holds a shearing box of a given metallicity, Stokes
    density (i.e. dust size distribution), gas viscosity, and a number of
    dust sizes considered. Linear modes can be calculated and stored in an
    HDF5 file.

    Attributes:
        param (dict): dictionary containing box parameters, metallicity,
            stokes range (minimum and maximum Stokes number), viscous_alpha
            (gas alpha viscosity), neglect_gas_viscosity (bool whether to
            neglect gas viscosity in gas momentum equations), n_dust (number
            of dust species), stokes_density_dict (dictionary containing
            additional information about the size distribution).
        direct_solver: Hermite spectral collocation solver to find eigenvalues
            and eigenvectors.
        equilibrium: EquilibriumBVP object, containing the equilibrium
            solution.
        filename (str): Name of HDF5 file to store eigenmodes. If set to
            None no output is provided.

    """
    ################
    # Constructors #
    ################

    def __init__(self,
                 metallicity,
                 stokes_range,
                 viscous_alpha,
                 stokes_density_dict=None,
                 neglect_gas_viscosity=True,
                 n_dust=1,
                 filename=None):
        """Setup direct solver, equilibrium and output file.

        Args:
            metallicity: dust mass fraction in box.
            stokes_range: minimum and maximum Stokes number.
            viscous_alpha: gas alpha viscosity.
            stokes_density_dict (optional): dictionary containing any
                additional information about the size distribution. Defaults
                to None, in which case an MRN size distribution is assumed.
            neglect_gas_viscosity (bool, optional): If True, gas viscosity is
                neglected in the gas momentum equations, but kept for
                calculating the level of dust diffusion. Defaults to True.
            n_dust (int, optional): Number of dust sizes (collocation points
                in size space). Defaults to 1, the monodisperse limit.
            filename (str, optional): Name of output HDF5 file. Defaults to
                None, in which case no output to file is done.

        """

        # Dictionary containing parameters
        self.param = {}

        # Physical setup dictionary
        self.param['metallicity'] = metallicity
        self.param['stokes_range'] = stokes_range
        self.param['viscous_alpha'] = viscous_alpha
        self.param['stokes_density_dict'] = stokes_density_dict
        self.param['neglect_gas_viscosity'] = bool(neglect_gas_viscosity)
        self.param['n_dust'] = n_dust

        # Create direct solver
        self.direct_solver = DirectSolver(interval=[-np.inf, np.inf],
                                          symmetry=None, basis='Hermite')

        # Solve for background state
        self.equilibrium = self.solve_background()

        # Create or check file
        self.filename = filename
        if self.filename is not None:
            self.check_main_group(self.filename)

    @classmethod
    def from_file(cls, filename):
        """Generate instance from previously saved HDF5 file.

        Args:
            filename (str): name of HDF5 file.

        """

        h5_file = h5.File(filename, 'a')

        g_main = h5_file['main']

        # Need at least these to create class instance
        required_keys = StratBox.required_parameters()

        # Check if all necessary parameters are present
        for k in required_keys:
            if k not in g_main.attrs:
                raise ValueError(f'{k} not present in hdf file')

        # Construct stokes_density_dict
        stokes_density_dict = g_main.attrs['stokes_density_dict']
        if stokes_density_dict == 'None':
            stokes_density_dict = None

        metallicity = g_main.attrs['metallicity']
        stokes_range = g_main.attrs['stokes_range']
        viscous_alpha = g_main.attrs['viscous_alpha']
        ngl = g_main.attrs['neglect_gas_viscosity']
        n_dust = g_main.attrs['n_dust']

        h5_file.close()

        # Create class
        return cls(metallicity,
                   stokes_range,
                   viscous_alpha,
                   stokes_density_dict=stokes_density_dict,
                   neglect_gas_viscosity=ngl,
                   n_dust=n_dust,
                   filename=filename)

    @staticmethod
    def required_parameters():
        """Return parameters required to setup instance.

        """
        # Return list of required parameters
        return ['metallicity',
                'stokes_range',
                'viscous_alpha',
                'neglect_gas_viscosity',
                'n_dust']

    def check_main_group(self, filename):
        """Create valid HDF5 file or check validity if it exists

        A valid HDF5 file contains a main group that has the attributes
        necessary to create a StratBox instance. This group is created is it
        does not exist, or checked against the param attribute.

        Args:
            filename (str): Name of HDF5 file.

        """

        h5_file = h5.File(filename, 'a')

        # Create main group and set attributes
        if "main" not in h5_file:
            g_main = h5_file.create_group('main')
            for key, item in self.param.items():
                # HDF does not accept None; convert to string
                if item is not None:
                    g_main.attrs[key] = item
                else:
                    g_main.attrs[key] = 'None'
        else:
            # Group exists: check all attributes
            g_main = h5_file['main']

            for key, item in self.param.items():
                if key not in g_main.attrs:
                    raise ValueError('Missing attribute in hdf file:', key)
                if np.atleast_1d(g_main.attrs[key])[0] != 'None':
                    if (g_main.attrs[key] != item).any():
                        raise ValueError('Attr has wrong value in hdf:', key)
                else:
                    if item is not None:
                        raise ValueError('Attr has wrong value in hdf:', key)

            for k in g_main.attrs:
                if k not in self.param:
                    raise ValueError('Extra attribute in hdf file:', k)

        h5_file.close()

    ###############################
    # Solve for equilibrium state #
    ###############################

    def stokes_density(self):
        """Create StokesDensity object.

        Create Stokes density object based on the stokes_range parameter and
        the stokes_density_dict dictionary.

        Returns:
            StokesDensity object.

        """

        # Create StokesDensity
        stokes_range = np.atleast_1d(self.param['stokes_range'])
        if self.param['n_dust'] == 1 and len(stokes_range) > 1:
            print('Warning: switching to monodisperse StokesDensity',
                  'because n_dust == 1')
            sigma = StokesDensity(self.param['stokes_range'][-1],
                                  self.param['stokes_density_dict'])
        else:
            sigma = StokesDensity(self.param['stokes_range'],
                                  self.param['stokes_density_dict'])

        return sigma

    def solve_background(self, background_resolution=1000):
        """Create equilibrium solution.

        Solve the boundary value problem for the equilibrium horizontal
        velocities.

        Args:
            background_resolution (int, optional): Number of collocation
            points to be used in BVP. Defaults to 1000.

        Returns:
            Equilibrium object.

        """
        equilibrium = Equilibrium(self.param['metallicity'],
                                  self.stokes_density(),
                                  self.param['n_dust'],
                                  viscous_alpha=self.param['viscous_alpha'])
        equilibrium.set_metallicity(self.param['metallicity'])

        ngl = self.param['neglect_gas_viscosity']
        equilibrium.solve_horizontal_velocities(background_resolution, neglect_gas_viscosity=ngl)
        return equilibrium

    ##########################
    # Solve for linear modes #
    ##########################

    def find_eigenvalues(self, wave_number_x, resolution, scale_factor=1,
                         sigma=None, n_eig=6):
        """Solve for linear modes.

        Solve the eigenvalue problem at a specific wave number. By default, a
        sparse eigensolver is used in shift-invert mode, to find the n_eig
        nearest eigenvalues around a specified complex number, sigma.

        Args:
            wave_number_x: Horizontal wave number to consider.
            resolution: Number of spectral collocation points.
            scale_factor (optional): Scaling for Hermite functions (see Boyd).
                Defaults to 1.
            sigma (optional): Complex number to find eigenvalues around.
            n_eig (int, optional): number of eigenvalues to try and find.
                Defaults to 6. Note that the number of values returned may
                differ, since non-converged eigenvalues are rejected.

        Returns:
            eigenvalues, eigenvectors, radius. The radius contains the maximum
            distance between eigenvalues found and sigma.

        """

        ngl = self.param['neglect_gas_viscosity']
        n_eq = 4 + 4*self.param['n_dust']

        degen = 1
        #if sparse_flag == True:
        #    degen = n_eig

        # Note: last three arguments are fed into matrix calculation
        eig, vec, rad = \
          self.direct_solver.safe_solve(resolution,
                                        L=scale_factor,
                                        n_eq=n_eq,
                                        sigma=sigma,
                                        n_eig=n_eig,
                                        degeneracy=degen,
                                        n_safe_levels=1,
                                        kx=wave_number_x,
                                        equilibrium=self.equilibrium,
                                        neglect_gas_viscosity=ngl)

        # Sort according to imaginary part: fastest growing first
        if len(eig) > 1:
            idx= np.argsort(-np.imag(eig))
            eig = eig[idx]
            vec = vec[idx]

        return eig, vec, rad

    #####################################
    # Saving/reading modes to/from file #
    #####################################

    def add_group_to_file(self, eig, vec, wave_number_x, scale_factor, label):
        """Add modes to HDF5 file.

        A group with the name contained in label is created under main/modes
        in the HDF5 file. Within this group, eigenvalue/eigenvector pairs is
        stored in groups named 0, 1, etc.

        Args:
            eig: list of eigenvalues.
            vec: list of eigenvectors.
            wave_number_x: horizontal wavenumber for all modes.
            scale_factor: scale factor (see Boyd) used for all modes.
            label (str): group name for HDF5 file.

        """

        if self.filename is None:
            print('Can not add group to file: no filename specified')
            return

        resolution = int(len(vec[0,:])/(4 + 4*self.param['n_dust']))

        h5_file = h5.File(self.filename, 'a')
        g_main = h5_file['main']

        if 'modes' not in g_main:
            print('Creating modes group')
            g_modes = g_main.create_group('modes')
        else:
            g_modes = g_main['modes']

        if label in g_modes:
            print('Warning: deleting group ', label)
            del g_modes[label]

        g_label = g_modes.create_group(label)
        g_label.attrs['kx'] = wave_number_x
        g_label.attrs['N'] = resolution
        g_label.attrs['L'] = scale_factor

        for i, e_val in enumerate(eig):
            g_mode = g_label.create_group(str(i))
            g_mode.attrs['eigenvalue'] = e_val

            print(f'Saving eigenvalue {e_val} under ', str(i))

            g_mode.create_dataset('eigenvector', data=vec[i])

        h5_file.close()

    def get_modes_in_label(self, label):
        """Return all modes contained under label in HDF5 file.

        Args:
            label (str): label to search under.

        Returns:
            list of mode names.

        """

        h5_file = h5.File(self.filename, 'r')

        g_mode = h5_file['main/modes/' + label]

        ret = []
        for i in g_mode.keys():
            ret.extend(i)

        h5_file.close()

        return ret

    def read_mode_from_file(self, label):
        """Read mode from HDF5 file.

        Args:
            label (str): mode name in HDF5 file, should live in main/modes.

        Returns:
            horizontal wave number, eigenvalue, eigenvector, z, space
            dependence of mode, space dependence of mode derivative. The
            latter three are only returned if present in the HDF5 file.

        """

        h5_file = h5.File(self.filename, 'r')

        g_mode = h5_file['main/modes/' + label]
        eigenvalue = g_mode.attrs['eigenvalue']

        vec = g_mode.get('eigenvector')[()]
        vertical_coordinate = None
        if 'z' in g_mode:
            vertical_coordinate = g_mode.get('z')[()]
        state = None
        if 'u' in g_mode:
            state = g_mode.get('u')[()]
        dstate = None
        if 'du' in g_mode:
            dstate = g_mode.get('du')[()]

        wave_number_x = g_mode.parent.attrs['kx']

        h5_file.close()

        return wave_number_x, eigenvalue, vec, \
          vertical_coordinate, state, dstate

    def compute_z(self, label, vertical_coordinate):
        """Compute spatial dependence of modes in HDF5 file under label.

        Reads eigenvector from HDF5 file, computes the spatial (z-)dependence
        of the modes and their derivatives, and puts these in the HDF5 file
        in the same group.

        Args:
            label (str): label in HDF5 file.
            vertical_coordinate (ndarray): array of z values.

        """

        if self.filename is None:
            print('Can not compute z: no filename specified')
            return

        h5_file = h5.File(self.filename, 'a')
        g_main = h5_file['main']

        if 'modes' not in g_main:
            print('Error: modes group does not exist')
        g_modes = g_main['modes']

        if label not in g_modes:
            print('Error: group does not exist:', label)
        g_label = g_modes[label]

        self.direct_solver.set_resolution(g_label.attrs['N'],
                                          L=g_label.attrs['L'],
                                          n_eq=4 + 4*self.param['n_dust'])

        for k in g_label.keys():
            g_mode = g_label[k]

            if 'z' in g_mode:
                print('Warning: replacing z')
                del g_mode['z']
            if 'u' in g_mode:
                print('Warning: replacing u')
                del g_mode['u']
            if 'du' in g_mode:
                print('Warning: replacing du')
                del g_mode['du']

            vec = g_mode.get('eigenvector')[()]

            state, dstate = \
              self.evaluate_velocity_form(vertical_coordinate, vec)

            g_mode.create_dataset('z', data=vertical_coordinate)
            g_mode.create_dataset('u', data=state)
            g_mode.create_dataset('du', data=dstate)

        h5_file.close()

    def evaluate_velocity_form(self, vertical_coordinate, vec):
        """Compute spatial dependence in velocity form.

        The eigenvalue problem is solved in momentum form, but often it is
        desirable to work with velocities rather than momenta. The spatial
        dependence of the mode is evaluated and converted to velocity form.

        Args:
            vertical_coordinate (ndarray): array of z values.
            vec: eigenvector.

        Returns:
            mode spatial dependence, mode z-derivative.

        """

        n_eq = 4 + 4*self.param['n_dust']

        state = self.direct_solver.evaluate(vertical_coordinate, vec, n_eq=n_eq)
        dstate = self.direct_solver.evaluate(vertical_coordinate,
                                             vec, n_eq=n_eq, k=1)

        rhog0 = self.equilibrium.gasdens(vertical_coordinate)
        dlogrhogdz = self.equilibrium.dlogrhogdz(vertical_coordinate)

        dstate[0,:] = (dstate[0,:] - state[0,:]*dlogrhogdz)/rhog0
        dstate[1,:] = (dstate[1,:] - state[1,:]*dlogrhogdz)/rhog0
        dstate[2,:] = (dstate[2,:] - state[2,:]*dlogrhogdz)/rhog0
        dstate[3,:] = (dstate[3,:] - state[3,:]*dlogrhogdz)/rhog0
        for i in range(0, self.param['n_dust']):
            rhod0 = self.equilibrium.sigma(vertical_coordinate)[i,:]
            dlogrhoddz = self.equilibrium.dlogsigmadz(vertical_coordinate)[i,:]

            dstate[4+4*i,:] = (dstate[4+4*i,:]- state[4+4*i,:]*dlogrhoddz)/rhod0
            dstate[5+4*i,:] = (dstate[5+4*i,:]- state[5+4*i,:]*dlogrhoddz)/rhod0
            dstate[6+4*i,:] = (dstate[6+4*i,:]- state[6+4*i,:]*dlogrhoddz)/rhod0
            dstate[7+4*i,:] = (dstate[7+4*i,:]- state[7+4*i,:]*dlogrhoddz)/rhod0

        state[0,:] = state[0,:]/rhog0
        state[1,:] = state[1,:]/rhog0
        state[2,:] = state[2,:]/rhog0
        state[3,:] = state[3,:]/rhog0
        for i in range(0, self.param['n_dust']):
            rhod0 = self.equilibrium.sigma(vertical_coordinate)[i,:]

            state[4+4*i,:] = state[4+4*i,:]/rhod0
            state[5+4*i,:] = state[5+4*i,:]/rhod0
            state[6+4*i,:] = state[6+4*i,:]/rhod0
            state[7+4*i,:] = state[7+4*i,:]/rhod0

        return state, dstate

    def get_total_dust_density(self, vertical_coordinate, state):
        """Compute total dust density.

        Compute the perturbation in total dust density.

        Args:
            vertical_coordinate (ndarray): array of z values.
            state: spatial dependence of perturbations.

        Returns:
            total dust density.

        """

        w_tau = self.equilibrium.tau*self.equilibrium.weights
        sigma = self.equilibrium.sigma(vertical_coordinate)

        # Calculate total dust density perturbation
        dust_rho = sigma[0,:]*state[4,:]*w_tau[0]
        dust_rho0 = sigma[0,:]*w_tau[0]
        for i in range(1, len(self.equilibrium.tau)):
            dust_rho = dust_rho + sigma[i,:]*state[4*(i+1),:]*w_tau[i]
            dust_rho0 = dust_rho0 + sigma[i,:]*w_tau[i]

        return dust_rho/dust_rho0
