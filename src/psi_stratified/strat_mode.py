import numpy as np
import h5py as h5

from .equilibrium import EquilibriumBVP
from .direct import DirectSolver
from .stokesdensity import StokesDensity

class StratBox():
    def __init__(self,
                 metallicity,
                 stokes_range,
                 viscous_alpha,
                 stokes_density_dict=None,
                 neglect_gas_viscosity=True):
        # Dictionary containing parameters
        self.param = {}

        # Physical setup dictionary
        self.param['metallicity'] = metallicity
        self.param['stokes_range'] = stokes_range
        self.param['viscous_alpha'] = viscous_alpha
        self.param['stokes_density_dict'] = stokes_density_dict
        self.param['neglect_gas_viscosity'] = bool(neglect_gas_viscosity)

        # Set number of dust sizes to invalid
        self.n_dust = -1

        # Create direct solver
        self.di = DirectSolver(interval=[-np.inf, np.inf],
                               symmetry=None, basis='Hermite')

        self.eig = []
        self.vec = []

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
        stokes_density_dict = StratBox.get_stokes_density_dict(g_main.attrs)

        # Create class
        return cls(g_main.attrs['metallicity'],
                   g_main.attrs['stokes_range'],
                   g_main.attrs['viscous_alpha'],
                   stokes_density_dict=stokes_density_dict,
                   neglect_gas_viscosity=g_main.attrs['neglect_gas_viscosity'])

    @staticmethod
    def required_parameters():
        # Return list of required parameters
        return ['metallicity',
                'stokes_range',
                'viscous_alpha',
                'neglect_gas_viscosity']

    @staticmethod
    def get_stokes_density_dict(attrs):
        required_keys = StratBox.required_parameters()

        # Construct stokes_density_dict
        stokes_density_dict = {}
        for k in attrs:
            stokes_density_dict[k] = attrs[k]
        # Remove required keys; anything left is in stokes_density_dict
        for k in required_keys:
            stokes_density_dict.pop(k)

        if len(stokes_density_dict) == 0:
            stokes_density_dict = None

        return stokes_density_dict

    def set_stokes_range(self, stokes_range):
        self.param['stokes_range'] = stokes_range
        # Create StokesDensity
        #self.sigma = StokesDensity(stokes_range,
        #                           self.param['stokes_density_dict'] )

        # Need to recalculate background
        self.solve_background(-1)

    def set_viscosity(self, viscous_alpha,
                      neglect_gas_viscosity=True):
        self.param['viscous_alpha'] = viscous_alpha
        self.param['neglect_gas_viscosity'] = bool(neglect_gas_viscosity)

        # Need to recalculate background
        self.solve_background(-1)

    def set_n_dust(self, n_dust):
        self.solve_background(n_dust)

    def solve_background(self, n_dust, N_back=1000):
        if self.n_dust != n_dust or n_dust < 0:
            if n_dust < 0:
                n_dust = self.n_dust

            # Create StokesDensity
            stokes_range = np.atleast_1d(self.param['stokes_range'])
            if n_dust == 1 and len(stokes_range) > 1:
                print('Warning: switching to monodisperse StokesDensity because n_dust == 1')
                self.sigma = StokesDensity(self.param['stokes_range'][-1],
                                           self.param['stokes_density_dict'])
            else:
                self.sigma = StokesDensity(self.param['stokes_range'],
                                           self.param['stokes_density_dict'])

            #print("Calculating background with {} dust size(s)".format(n_dust))
            self.n_dust = n_dust
            metallicity = self.param['metallicity']
            self.eq = EquilibriumBVP(self.sigma,
                                     self.n_dust,
                                     viscous_alpha=self.param['viscous_alpha'],
                                     dust_to_gas_ratio=100*metallicity)
            self.eq.set_metallicity(metallicity)

            ngl = self.param['neglect_gas_viscosity']
            self.eq.solve(N_back, neglect_gas_viscosity=ngl)

    def find_eigenvalues(self, wave_number_x, N, L=1, n_dust=1,
                         sparse_flag=False, sigma=None, n_eig=6,
                         use_PETSc=False):
        if self.sigma.stokes_min is None and n_dust > 1:
            print("Warning: setting n_dust=1 because monodisperse!")
            n_dust = 1

        self.solve_background(n_dust)

        # Find all eigenvalues if n_eig < 0
        #if n_eig < 0:
        #    n_eig = 4*(n_dust + 1)*N

        #print('Finding eigenvalues...')

        self.kx = wave_number_x
        self.L = L
        self.N = N
        self.n_dust = n_dust
        ngl = self.param['neglect_gas_viscosity']

        degen = 1
        #if sparse_flag == True:
        #    degen = n_eig
        self.eig, self.vec = \
          self.di.safe_solve(N,
                             L=L,
                             n_eq=4 + 4*self.n_dust,
                             sparse_flag=sparse_flag,
                             sigma=sigma, n_eig=n_eig,
                             use_PETSc=use_PETSc,
                             degeneracy=degen,
                             kx=wave_number_x,
                             equilibrium=self.eq,
                             neglect_gas_viscosity=ngl)

    def select_growing_eigenvalues(self):
        eig_select = []
        vec_select = []
        if len(self.eig) > 0:
            for n, v in enumerate(self.vec):
                if np.imag(self.eig[n]) > 0:
                    eig_select.append(self.eig[n])
                    vec_select.append(self.vec[n])

        self.eig = eig_select
        self.vec = vec_select

    def read_from_file(self, filename, wave_number_x, N, L, n_dust):
        hf = h5.File(filename, 'r')

        k_string = str(int(wave_number_x))

        if "main" not in hf:
            raise RunTimeError("main group not in hdf file")
        g_main = hf['main']

        # Check if attributes file match current StratBox
        for k in StratBox.required_parameters():
            if g_main.attrs[k] != self.param[k]:
                raise ValueError('Parameter {} not equal in hdf file!')
        stokes_dict_hdf = StratBox.get_stokes_density_dict(g_main.attrs)
        stokes_dict_par = self.param['stokes_density_dict']
        if stokes_dict_hdf is None:
            if stokes_dict_par is not None:
                raise ValueError('missing stokes_density_dict in hdf file')
        elif len(stokes_dict_hdf) !=len(stokes_dict_par):
            raise ValueError('different stokes_density_dict in hdf file')
        else:
            for k in stokes_dict_hdf:
                if stokes_dict_hdf[k] != stokes_dict_par[k]:
                    raise ValueError('diff stokes_density_dict in hdf file')

        # Work our way through groups to the data
        if k_string not in g_main:
            raise RuntimeError("kx group not in hdf file")
        g_kx = g_main[k_string]
        if str(N) not in g_kx:
            raise RuntimeError("N group not in hdf file")
        g_N = g_kx[str(N)]
        if str(n_dust) not in g_N:
            raise RuntimeError("n_dust group not in hdf file")
        g_n_dust = g_N[str(n_dust)]
        if str(n_dust) not in g_N:
            raise RuntimeError("n_dust group not in hdf file")
        Lstring = '{:.2e}'.format(L)
        if Lstring not in g_n_dust:
            raise RuntimeError("L group not in hdf file ", Lstring)
        g_L = g_n_dust[Lstring]

        # Get data
        self.eig = g_L.get('eigenvalues')[()]
        self.vec = g_L.get('eigenvectors')[()]

        # Set background
        self.solve_background(n_dust)
        self.kx = wave_number_x
        self.L = L

        # Correct resolution for direct solver
        self.di.set_resolution(N, L=L, n_eq=4+4*n_dust)

    def save(self, filename):
        hf = h5.File(filename, 'a')

        # Create main group and set attributes
        if "main" not in hf:
            print('Creating main group')
            g_main = hf.create_group('main')
            for k in self.param.keys():
                # HDF does not accept None; convert to string
                if self.param[k] is not None:
                    g_main.attrs[k] = self.param[k]
                else:
                    g_main.attrs[k] = 'None'
        else:
            print('Main group exists')

            # Group exists: check all attributes
            g_main = hf['main']

            for k in self.param.keys():
                if k not in g_main.attrs:
                    raise ValueError('Missing attribute in hdf file:', k)
                if g_main.attrs[k] != 'None':
                    if g_main.attrs[k] != self.param[k]:
                        raise ValueError('Attr has wrong value in hdf file:', k)
                else:
                    if self.param[k] is not None:
                        raise ValueError('Attr has wrong value in hdf file:', k)

            for k in g_main.attrs:
                if k not in self.param:
                    raise ValueError('Extra attribute in hdf file:', k)

        if len(np.atleast_1d(self.eig)) > 0:
            # Compute N from length of eigenvector
            n_eq=4 + 4*self.n_dust
            N = int(len(self.vec[0])/n_eq)

            k_string = str(int(self.kx))

            if k_string not in g_main:
                print('Creating new k group')
                g_kx = g_main.create_group(k_string)
            else:
                g_kx = g_main[k_string]

            if str(N) not in g_kx:
                print('Creating new N group')
                g_N = g_kx.create_group(str(N))
            else:
                g_N = g_kx[str(N)]

            if str(self.n_dust) not in g_N:
                print('Creating new n_dust group')
                g_n_dust = g_N.create_group(str(self.n_dust))
            else:
                g_n_dust = g_N[str(self.n_dust)]
                #print('Warning: replacing eigenvalues and eigenvectors!')
                #del g_n_dust['eigenvalues']
                #del g_n_dust['eigenvectors']

            Lstring = '{:.2e}'.format(self.L)
            if Lstring not in g_n_dust:
                print('Creating new L group {}'.format(Lstring))
                g_L = g_n_dust.create_group(Lstring)
            else:
                g_L = g_n_dust[Lstring]
                print('Warning: replacing eigenvalues and eigenvectors!')
                del g_L['eigenvalues']
                del g_L['eigenvectors']


            g_L.create_dataset('eigenvalues', data=self.eig)
            g_L.create_dataset('eigenvectors', data=self.vec)

        hf.close()

    def evaluate_velocity_form(self, z, vec, k=0):
        n_eq = 4 + 4*self.n_dust

        if k == 0:
            u = self.di.evaluate(z, vec, n_eq=n_eq)

            rhog0 = self.eq.gasdens(z)
            rhod0 = self.eq.sigma(z)[0,:]

            u[0,:] = u[0,:]/rhog0
            u[1,:] = u[1,:]/rhog0
            u[2,:] = u[2,:]/rhog0
            u[3,:] = u[3,:]/rhog0
            for n in range(0, self.n_dust):
                rhod0 = self.eq.sigma(z)[n,:]

                u[4+4*n,:] = u[4+4*n,:]/rhod0
                u[5+4*n,:] = u[5+4*n,:]/rhod0
                u[6+4*n,:] = u[6+4*n,:]/rhod0
                u[7+4*n,:] = u[7+4*n,:]/rhod0

        elif k == 1:
            u = self.di.evaluate(z, vec, n_eq=n_eq)
            du = self.di.evaluate(z, vec, n_eq=n_eq, k=1)

            rhog0 = self.eq.gasdens(z)
            dlogrhogdz = self.eq.dlogrhogdz(z)

            u[0,:] = (du[0,:] - u[0,:]*dlogrhogdz)/rhog0
            u[1,:] = (du[1,:] - u[1,:]*dlogrhogdz)/rhog0
            u[2,:] = (du[2,:] - u[2,:]*dlogrhogdz)/rhog0
            u[3,:] = (du[3,:] - u[3,:]*dlogrhogdz)/rhog0
            for n in range(0, self.n_dust):
                rhod0 = self.eq.sigma(z)[n,:]
                dlogrhoddz = self.eq.dlogsigmadz(z)[n,:]

                u[4+4*n,:] = (du[4+4*n,:] - u[4+4*n,:]*dlogrhoddz)/rhod0
                u[5+4*n,:] = (du[5+4*n,:] - u[5+4*n,:]*dlogrhoddz)/rhod0
                u[6+4*n,:] = (du[6+4*n,:] - u[6+4*n,:]*dlogrhoddz)/rhod0
                u[7+4*n,:] = (du[7+4*n,:] - u[7+4*n,:]*dlogrhoddz)/rhod0
        else:
            raise RunTimeError('Not implemented')

        return u
