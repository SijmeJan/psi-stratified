# -*- coding: utf-8 -*-
"""Module containing ShearingBoxParam class.
"""

import numpy as np

class ShearingBoxParam():
    """Class containing shearing box parameters

    Basically a dictionary plus a check function to see if all necessary keys
    are present and if all values make sense, so that a valid instance of
    ShearingBox can be created.

    Attributes:
        param: dictionary containing keys metallicity, stokes_range,
        viscous_alpha, stokes_density_dict, neglect_gas_viscosity, n_dust.
    """
    def __init__(self):
        self.param = {
            'stokes_density_dict' : None,
            'neglect_gas_viscosity' : True,
            'n_dust' : 1
        }

    def check(self):
        """Check if all parameters are present and valid"""

        # Check if all keys are present in self.param
        if 'metallicity' not in self.param:
            raise KeyError('ShearingBoxParam.param should contain metallicity key')
        if 'stokes_range' not in self.param:
            raise KeyError('ShearingBoxParam.param should contain stokes_range key')
        if 'viscous_alpha' not in self.param:
            raise KeyError('ShearingBoxParam.param should contain viscous_alpha key')
        # These are set by default, so if they are not present something is seriously wrong
        if 'stokes_density_dict' not in self.param:
            raise KeyError('ShearingBoxParam.param should contain stokes_density_dict key')
        if 'neglect_gas_viscosity' not in self.param:
            raise KeyError('ShearingBoxParam.param should contain neglect_gas_viscosity key')
        if 'n_dust' not in self.param:
            raise KeyError('ShearingBoxParam.param should contain n_dust key')

        # Check if values are valid
        if self.param['metallicity'] < 0:
            raise ValueError('Metallicity needs to be >= 0')
        if len(self.param['stokes_range']) != 2:
            raise ValueError('stokes_range needs to have 2 elements')
        if np.min(self.param['stokes_range']) <= 0:
            raise ValueError('stokes_range entries need to be > 0')
        if self.param['stokes_range'][0] >= self.param['stokes_range'][1]:
            raise ValueError('stokes_range elements must be in increasing order')
        if self.param['viscous_alpha'] <= 0:
            raise ValueError('viscous_alpha needs to be > 0')
        if not isinstance(self.param['neglect_gas_viscosity'], (bool, np.bool_)):
            raise TypeError('neglect_gas_viscosity should be boolean')
        if not isinstance(self.param['n_dust'], (int, np.int64)):
            raise TypeError('n_dust should be int')
        if self.param['n_dust'] <= 0:
            raise ValueError('n_dust must be >= 0')

class ModeParam():
    """Class containing mode parameters

    Basically a dictionary plus a check function to see if all necessary keys
    are present and if all values make sense.

    Attributes:
        param: dictionary containing keys wave_number_x, n_coll, scale_l.
    """
    def __init__(self):
        self.param = {
            'scale_l' : 1
        }

    def check(self):
        """Check if all parameters are present and valid"""

        # Check if all keys are present in self.param
        if 'wave_number_x' not in self.param:
            raise KeyError('ModeParam.param should contain wave_number_x key')
        if 'n_coll' not in self.param:
            raise KeyError('ModeParam.param should contain n_coll key')
        if 'scale_l' not in self.param:
            raise KeyError('ModeParam.param should contain scale_l key')

        # Check if values are valid
        if self.param['scale_l'] <= 0:
            raise ValueError('scale_l needs to be > 0')
        if not isinstance(self.param['n_coll'], (int, np.int64)):
            raise TypeError('n_coll should be int')
        if self.param['n_coll'] <= 0:
            raise ValueError('n_coll must be >= 0')
