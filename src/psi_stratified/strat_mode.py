import numpy as np
import h5py as h5

from .equilibrium import EquilibriumBVP
from .direct import DirectSolver
from .stokesdensity import StokesDensity

class StratBox():
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
        self.di = DirectSolver(interval=[-np.inf, np.inf],
                               symmetry=None, basis='Hermite')

        # Solve for background state
        self.eq = self.solve_background()

        # Create or check file
        self.filename = filename
        if self.filename is not None:
            self.check_main_group(self.filename)

    @classmethod
    def from_file(cls, filename):
        hf = h5.File(filename, 'a')

        g_main = hf['main']

        # Need at least these to create class instance
        required_keys = StratBox.required_parameters()

        # Check if all necessary parameters are present
        for k in required_keys:
            if k not in g_main.attrs:
                raise ValueError('{} not present in hdf file'.format(k))

        # Construct stokes_density_dict
        stokes_density_dict = g_main.attrs['stokes_density_dict']
        if stokes_density_dict == 'None':
            stokes_density_dict = None

        metallicity = g_main.attrs['metallicity']
        stokes_range = g_main.attrs['stokes_range']
        viscous_alpha = g_main.attrs['viscous_alpha']
        ngl = g_main.attrs['neglect_gas_viscosity']
        n_dust = g_main.attrs['n_dust']

        hf.close()

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
        # Return list of required parameters
        return ['metallicity',
                'stokes_range',
                'viscous_alpha',
                'neglect_gas_viscosity',
                'n_dust']

    def check_main_group(self, filename):
        hf = h5.File(filename, 'a')

        # Create main group and set attributes
        if "main" not in hf:
            #print('Creating main group')
            g_main = hf.create_group('main')
            for k in self.param.keys():
                # HDF does not accept None; convert to string
                if self.param[k] is not None:
                    g_main.attrs[k] = self.param[k]
                else:
                    g_main.attrs[k] = 'None'
        else:
            #print('Main group exists')

            # Group exists: check all attributes
            g_main = hf['main']

            for k in self.param.keys():
                if k not in g_main.attrs:
                    raise ValueError('Missing attribute in hdf file:', k)
                if np.atleast_1d(g_main.attrs[k])[0] != 'None':
                    #print(g_main.attrs[k], self.param[k])
                    if (g_main.attrs[k] != self.param[k]).any():
                        raise ValueError('Attr has wrong value in hdf file:', k)
                else:
                    if self.param[k] is not None:
                        raise ValueError('Attr has wrong value in hdf file:', k)

            for k in g_main.attrs:
                if k not in self.param:
                    raise ValueError('Extra attribute in hdf file:', k)

        hf.close()

    #######################################
    # Methods for changing box parameters #
    #######################################

    #def set_stokes_range(self, stokes_range):
    #    self.param['stokes_range'] = stokes_range
    #    # Need to recalculate background
    #    self.solve_background(-1)

    #def set_viscosity(self, viscous_alpha,
    #                  neglect_gas_viscosity=True):
    #    self.param['viscous_alpha'] = viscous_alpha
    #    self.param['neglect_gas_viscosity'] = bool(neglect_gas_viscosity)

    #    # Need to recalculate background
    #    self.solve_background(-1)

    #def set_n_dust(self, n_dust):
    #    # Need to recalculate background
    #    self.solve_background(n_dust)

    ###############################
    # Solve for equilibrium state #
    ###############################

    def stokes_density(self):
        # Create StokesDensity
        stokes_range = np.atleast_1d(self.param['stokes_range'])
        if self.param['n_dust'] == 1 and len(stokes_range) > 1:
            print('Warning: switching to monodisperse StokesDensity because n_dust == 1')
            sigma = StokesDensity(self.param['stokes_range'][-1],
                                  self.param['stokes_density_dict'])
        else:
            sigma = StokesDensity(self.param['stokes_range'],
                                  self.param['stokes_density_dict'])

        return sigma

    def solve_background(self, N_back=1000):
        eq = EquilibriumBVP(self.stokes_density(),
                            self.param['n_dust'],
                            viscous_alpha=self.param['viscous_alpha'],
                            dust_to_gas_ratio=100*self.param['metallicity'])
        eq.set_metallicity(self.param['metallicity'])

        ngl = self.param['neglect_gas_viscosity']
        eq.solve(N_back, neglect_gas_viscosity=ngl)

        return eq

    ##########################
    # Solve for linear modes #
    ##########################

    def find_eigenvalues(self, wave_number_x, N, L=1,
                         sparse_flag=False, sigma=None, n_eig=6,
                         n_safe_levels=1, use_PETSc=False, label=None):
        # THIS IS BAD
        #self.kx = wave_number_x
        #self.L = L
        #self.N = N
        ngl = self.param['neglect_gas_viscosity']
        n_eq = 4 + 4*self.param['n_dust']

        degen = 1
        #if sparse_flag == True:
        #    degen = n_eig
        eig, vec, rad = \
          self.di.safe_solve(N,
                             L=L,
                             n_eq=n_eq,
                             sigma=sigma, n_eig=n_eig,
                             degeneracy=degen,
                             n_safe_levels=n_safe_levels,
                             kx=wave_number_x,              # kwarg for M
                             equilibrium=self.eq,           # kwarg for M
                             neglect_gas_viscosity=ngl)     # kwarg for M

        # Sort according to imaginary part: fastest growing first
        if len(eig) > 1:
            idx= np.argsort(-np.imag(eig))
            eig = eig[idx]
            vec = vec[idx]

        if self.filename is not None and label is not None:
            self.add_group_to_file(eig. vec, wave_number_x, N, L, label)

        return eig, vec, rad

    #def select_growing_eigenvalues(self):
    #    eig_select = []
    #    vec_select = []
    #    if len(self.eig) > 0:
    #        for n, v in enumerate(self.vec):
    #            if np.imag(self.eig[n]) > 0:
    #                eig_select.append(self.eig[n])
    #                vec_select.append(self.vec[n])

    #    self.eig = eig_select
    #    self.vec = vec_select

    #####################################
    # Saving/reading modes to/from file #
    #####################################

    def add_group_to_file(self, eig, vec, kx, L, label):
        if self.filename is None:
            print('Can not add group to file: no filename specified')
            return

        N = int(len(vec[0,:])/(4 + 4*self.param['n_dust']))

        hf = h5.File(self.filename, 'a')
        g_main = hf['main']

        if 'modes' not in g_main:
            print('Creating modes group')
            g_modes = g_main.create_group('modes')
        else:
            g_modes = g_main['modes']

        if label in g_modes:
            print('Warning: deleting group ', label)
            del g_modes[label]

        g_label = g_modes.create_group(label)
        g_label.attrs['kx'] = kx
        g_label.attrs['N'] = N
        g_label.attrs['L'] = L

        for i in range(0, len(eig)):
            g = g_label.create_group(str(i))
            g.attrs['eigenvalue'] = eig[i]

            print('Saving eigenvalue {} under '.format(eig[i]), str(i))

            g.create_dataset('eigenvector', data=vec[i])

        hf.close()

    def get_modes_in_label(self, label):
        hf = h5.File(self.filename, 'r')

        g = hf['main/modes/' + label]

        ret = []
        for i in g.keys():
            ret.extend(i)

        hf.close()

        return ret


    def read_mode_from_file(self, label):
        hf = h5.File(self.filename, 'r')

        g = hf['main/modes/' + label]
        eigenvalue = g.attrs['eigenvalue']

        vec = g.get('eigenvector')[()]
        z = None
        if 'z' in g:
            z = g.get('z')[()]
        u = None
        if 'u' in g:
            u = g.get('u')[()]
        du = None
        if 'du' in g:
            du = g.get('du')[()]

        kx = g.parent.attrs['kx']

        hf.close()

        return kx, eigenvalue, vec, z, u, du

    def compute_z(self, label, z):
        if self.filename is None:
            print('Can not compute z: no filename specified')
            return

        hf = h5.File(self.filename, 'a')
        g_main = hf['main']

        if 'modes' not in g_main:
            print('Error: modes group does not exist')
        g_modes = g_main['modes']

        if label not in g_modes:
            print('Error: group does not exist:', label)
        g_label = g_modes[label]

        self.di.set_resolution(g_label.attrs['N'],
                               L=g_label.attrs['L'],
                               n_eq=4 + 4*self.param['n_dust'])

        for k in g_label.keys():
            g = g_label[k]

            if 'z' in g:
                print('Warning: replacing z')
                del g['z']
            if 'u' in g:
                print('Warning: replacing u')
                del g['u']
            if 'du' in g:
                print('Warning: replacing du')
                del g['du']

            vec = g.get('eigenvector')[()]

            u, du = self.evaluate_velocity_form(z, vec)

            g.create_dataset('z', data=z)
            g.create_dataset('u', data=u)
            g.create_dataset('du', data=du)

        hf.close()

    def evaluate_velocity_form(self, z, vec):
        n_eq = 4 + 4*self.param['n_dust']

        u = self.di.evaluate(z, vec, n_eq=n_eq)
        du = self.di.evaluate(z, vec, n_eq=n_eq, k=1)

        rhog0 = self.eq.gasdens(z)
        dlogrhogdz = self.eq.dlogrhogdz(z)

        du[0,:] = (du[0,:] - u[0,:]*dlogrhogdz)/rhog0
        du[1,:] = (du[1,:] - u[1,:]*dlogrhogdz)/rhog0
        du[2,:] = (du[2,:] - u[2,:]*dlogrhogdz)/rhog0
        du[3,:] = (du[3,:] - u[3,:]*dlogrhogdz)/rhog0
        for n in range(0, self.param['n_dust']):
            rhod0 = self.eq.sigma(z)[n,:]
            dlogrhoddz = self.eq.dlogsigmadz(z)[n,:]

            du[4+4*n,:] = (du[4+4*n,:] - u[4+4*n,:]*dlogrhoddz)/rhod0
            du[5+4*n,:] = (du[5+4*n,:] - u[5+4*n,:]*dlogrhoddz)/rhod0
            du[6+4*n,:] = (du[6+4*n,:] - u[6+4*n,:]*dlogrhoddz)/rhod0
            du[7+4*n,:] = (du[7+4*n,:] - u[7+4*n,:]*dlogrhoddz)/rhod0

        u[0,:] = u[0,:]/rhog0
        u[1,:] = u[1,:]/rhog0
        u[2,:] = u[2,:]/rhog0
        u[3,:] = u[3,:]/rhog0
        for n in range(0, self.param['n_dust']):
            rhod0 = self.eq.sigma(z)[n,:]

            u[4+4*n,:] = u[4+4*n,:]/rhod0
            u[5+4*n,:] = u[5+4*n,:]/rhod0
            u[6+4*n,:] = u[6+4*n,:]/rhod0
            u[7+4*n,:] = u[7+4*n,:]/rhod0

        return u, du

    def get_total_dust_density(self, z, u):
        wt = self.eq.tau*self.eq.weights

        # Calculate total dust density perturbation
        dust_rho = self.eq.sigma(z)[0,:]*u[4,:]*wt[0]
        dust_rho0 = self.eq.sigma(z)[0,:]*wt[0]
        for i in range(1, len(self.eq.tau)):
            dust_rho = dust_rho + \
              self.eq.sigma(z)[i,:]*u[4*(i+1),:]*wt[i]
            dust_rho0 = dust_rho0 + \
              self.eq.sigma(z)[i,:]*wt[i]

        return dust_rho/dust_rho0
