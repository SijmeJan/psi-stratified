import numpy as np

def dust_density(u, sigma0, tau, weights):
    dust_rho = sigma0[0,:]*u[4,:]*tau[0]*weights[0]
    dust_rho0 = sigma0[0,:]*tau[0]*weights[0]
    for i in range(1, len(tau)):
        dust_rho = dust_rho + sigma0[i,:]*u[4*(i+1),:]*tau[i]*weights[i]
        dust_rho0 = dust_rho0 + sigma0[i,:]*tau[i]*weights[i]
    return dust_rho/dust_rho0

def norm_factor_dust_density(u, sigma0, tau, weights):
    dust_rho = dust_density(u, sigma0, tau, weights)

    i_real = np.argmax(np.abs(np.real(dust_rho)))
    i_imag = np.argmax(np.abs(np.imag(dust_rho)))
    norm_fac = np.real(dust_rho[i_real])
    if np.abs(np.real(dust_rho[i_real])) < np.abs(np.imag(dust_rho[i_imag])):
        norm_fac = np.imag(dust_rho[i_imag])

    return norm_fac
