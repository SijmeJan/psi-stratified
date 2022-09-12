# -*- coding: utf-8 -*-
"""Various tools.
"""
import numpy as np

def dust_density(pert, sigma0, stokes, weights):
    """Calculate dust density perturnation

    Args:
        pert: perturbation eigenvectors
        sigma0: equilibrium Stokes density
        stokes: Stokes numbers
        weights: quadrature weights in St space

    Returns:
        dust density perturbation, normalized by equilibrium
    """
    dust_rho = sigma0[0,:]*pert[4,:]*stokes[0]*weights[0]
    dust_rho0 = sigma0[0,:]*stokes[0]*weights[0]
    for i in range(1, len(stokes)):
        dust_rho = dust_rho + sigma0[i,:]*pert[4*(i+1),:]*stokes[i]*weights[i]
        dust_rho0 = dust_rho0 + sigma0[i,:]*stokes[i]*weights[i]
    return dust_rho/dust_rho0

def norm_factor_dust_density(pert, sigma0, stokes, weights):
    """Calculate normalization factor for dust density perturbation

    Args:
        pert: perturbation eigenvectors
        sigma0: equilibrium Stokes density
        stokes: Stokes numbers
        weights: quadrature weights in St space

    Returns:
        normalization factor
    """
    dust_rho = dust_density(pert, sigma0, stokes, weights)

    i_real = np.argmax(np.abs(np.real(dust_rho)))
    i_imag = np.argmax(np.abs(np.imag(dust_rho)))
    norm_fac = np.real(dust_rho[i_real])
    if np.abs(np.real(dust_rho[i_real])) < np.abs(np.imag(dust_rho[i_imag])):
        norm_fac = np.imag(dust_rho[i_imag])

    return norm_fac
