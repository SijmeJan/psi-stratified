# -*- coding: utf-8 -*-
"""Module dealing with eigenvalue finder
"""

import numpy as np

def unique_within_tol_complex(vec, tol=1e-12):
    """Brute force finding unique complex values"""
    if vec.size == 0:
        return vec

    sel = np.ones(len(vec), dtype=bool)

    for i in range(0, len(vec)-1):
        dist = np.abs(vec[i+1:] - vec[i]).tolist()
        if np.min(dist) < tol:
            sel[i] = False

    return vec[sel], sel

class ModeFinder():
    """Class for finding modes in regions of parameter space

    Args:
        strat_box: StratBox object
    """
    def __init__(self, strat_box):
        self.sbx = strat_box

    def find_optimal_scale(self, wave_number_x, n_coll, n_l=10):
        """Find optimal scaling factor"""
        scale_l = np.logspace(-3, 0, n_l)
        n_found = 0*scale_l

        for i in range(0, n_l):
            self.sbx.find_eigenvalues(wave_number_x=wave_number_x,
                                      N=n_coll,
                                      L=scale_l[i],
                                      sparse_flag=True,
                                      n_safe_levels=2,
                                      use_PETSc=True,
                                      sigma=0 + 0.001j,
                                      n_eig=-1)
            n_found[i] = len(self.sbx.eig)
            print(i, scale_l[i], n_found[i])

        return scale_l[np.argmax(n_found)]

    def find_growing_at_real_part(self, real_part,
                                  imag_range,
                                  mode_param,
                                  n_eig=10,
                                  flip_real_imag_flag=False):
        """Search around given real part for growing modes."""
        imag_part = imag_range[0]

        e_val = []
        e_vec = []

        centres = []
        radii = []

        while imag_part < imag_range[1]:
            # Search around this target
            sigma = real_part + 1j*imag_part

            # Flip if tracking real line
            if flip_real_imag_flag:
                sigma = imag_part + 1j*real_part

            eig, vec, rad = \
              self.sbx.find_eigenvalues(mode_param, sigma=sigma, n_eig=n_eig)

            centres.append(sigma)
            radii.append(rad)

            print(imag_part, rad, len(e_val), flush=True)
            # sbx.rad: maximum distance between sigma and any eigenvalue found
            imag_part = imag_part + rad

            if len(eig) > 0:
                e_val.extend(eig.tolist())
                e_vec.extend(vec.tolist())

        # Prune to unique eigenvalues
        if len(e_val) > 0:
            e_val, sel = unique_within_tol_complex(np.asarray(e_val),
                                                   tol=1.0e-3)
            e_vec = np.asarray(e_vec)[sel,:]

        return e_val, e_vec, centres, radii
