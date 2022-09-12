# -*- coding: utf-8 -*-
"""Module dealing with energy decomposition.
"""

import numpy as np

from .tools import norm_factor_dust_density

def energy_decomposition(z_coord, s_box, eig, k_x, pert, d_pert):
    """Perform energy decomposition
    """
    # Equilibrium structure
    stokes_numbers = s_box.equilibrium.stokes_numbers
    #rhog0 = s_box.equilibrium.eq_rhog.evaluate(z_coord)
    rhod0 = s_box.equilibrium.eq_sigma.evaluate(z_coord)
    mu0 = s_box.equilibrium.eq_mu.evaluate(z_coord)
    #dlogrhogdz = s_box.equilibrium.eq_rhog.log_deriv(z_coord)
    #dlogsigmadz = s_box.equilibrium.eq_sigma.log_deriv(z_coord)
    gas_vx0, gas_vy0, gas_vz0, dust_ux0, dust_uy0, dust_uz0 = \
      s_box.equilibrium.evaluate_velocities(z_coord, k=0)
    dgas_vx0, dgas_vy0, dgas_vz0, ddust_ux0, ddust_uy0, ddust_uz0 = \
      s_box.equilibrium.evaluate_velocities(z_coord, k=1)
    #d2gas_vx0, d2gas_vy0, d2gas_vz0, d2dust_ux0, d2dust_uy0, d2dust_dust_uz0 = \
    #  s_box.equilibrium.evaluate_velocities(z_coord, k=2)

    # Normalize to maximum dust density perturbation
    norm_fac = norm_factor_dust_density(pert, rhod0,
                                        stokes_numbers,
                                        s_box.equilibrium.weights)
    pert = pert/norm_fac
    d_pert = d_pert/norm_fac

    rhog = pert[0,:]
    gas_vx = pert[1,:]
    gas_vy = pert[2,:]
    gas_vz = pert[3,:]
    rhod = pert[4,:]
    dust_ux = pert[5,:]
    dust_uy = pert[6,:]
    dust_uz = pert[7,:]

    drhog = d_pert[0,:]
    #dgas_vx = d_pert[1,:]
    #dgas_vy = d_pert[2,:]
    #dgas_vz = d_pert[3,:]
    #drhod = d_pert[4,:]
    ddust_ux = d_pert[5,:]
    ddust_uy = d_pert[6,:]
    ddust_uz = d_pert[7,:]

    stkw = s_box.equilibrium.stokes_numbers[0]*s_box.equilibrium.weights[0]
    u_1 = -(stkw*mu0[0,:]*ddust_ux0[0,:]*np.real(dust_uz*np.conjugate(dust_ux)) + \
           dgas_vx0*np.real(gas_vz*np.conjugate(gas_vx))) - \
          4*(stkw*mu0[0,:]*ddust_uy0[0,:]*np.real(dust_uz*np.conjugate(dust_uy)) + \
             dgas_vy0*np.real(gas_vz*np.conjugate(gas_vy))) - \
          stkw*mu0[0,:]*ddust_uz0[0,:]*np.abs(dust_uz)*np.abs(dust_uz)

    u_2 = -stkw*mu0[0,:]*dust_uz0[0,:]*np.real(ddust_ux*np.conjugate(dust_ux) + \
                                      4*ddust_uy*np.conjugate(dust_uy) + \
                                      ddust_uz*np.conjugate(dust_uz))

    u_3 = k_x*np.imag(rhog*np.conjugate(gas_vx)) - \
         np.real(drhog*np.conjugate(gas_vz))

    # q_m = mu/mu0
    q_m = rhod - rhog

    u_4 = -stkw*mu0[0,:]*((gas_vx0 - dust_ux0[0,:])*np.real(q_m*np.conjugate(gas_vx)) + \
               4*(gas_vy0 - dust_uy0[0,:])*np.real(q_m*np.conjugate(gas_vy)) + \
               np.abs(gas_vx - dust_ux)*np.abs(gas_vx - dust_ux) + \
               4*np.abs(gas_vy - dust_uy)*np.abs(gas_vy - dust_uy) + \
               np.abs(gas_vz - dust_uz)*np.abs(gas_vz - dust_uz))/stokes_numbers[0]


    u_5 = stkw*mu0[0,:]*dust_uz0[0,:]*np.real(q_m*np.conjugate(gas_vz))/stokes_numbers[0]

    u_tot = stkw*mu0[0,:]*(np.abs(dust_ux)*np.abs(dust_ux) + \
                4*np.abs(dust_uy)*np.abs(dust_uy) + \
                np.abs(dust_uz)*np.abs(dust_uz)) + \
      np.abs(gas_vx)*np.abs(gas_vx) + \
      4*np.abs(gas_vy)*np.abs(gas_vy) + \
      np.abs(gas_vz)*np.abs(gas_vz)

    for i in range(1, s_box.equilibrium.n_dust):
        rhod = pert[4 + 4*i,:]
        dust_ux = pert[4 + 4*i + 1,:]
        dust_uy = pert[4 + 4*i + 2,:]
        dust_uz = pert[4 + 4*i + 3,:]
        ddust_ux = d_pert[4 + 4*i + 1,:]
        ddust_uy = d_pert[4 + 4*i + 2,:]
        ddust_uz = d_pert[4 + 4*i + 3,:]

        stkw = s_box.equilibrium.stokes_numbers[i]*s_box.equilibrium.weights[i]
        u_1 += -stkw*mu0[i,:]*ddust_ux0[i,:]*np.real(dust_uz*np.conjugate(dust_ux)) - \
          4*stkw*mu0[i,:]*ddust_uy0[i,:]*np.real(dust_uz*np.conjugate(dust_uy)) - \
          stkw*mu0[i,:]*ddust_uz0[i,:]*np.abs(dust_uz)*np.abs(dust_uz)

        u_2 += -stkw*mu0[i,:]*dust_uz0[i,:]*np.real(ddust_ux*np.conjugate(dust_ux) + \
                                           4*ddust_uy*np.conjugate(dust_uy) + \
                                           ddust_uz*np.conjugate(dust_uz))

        q_m = rhod - rhog
        u_4 += -stkw*mu0[i,:]*((gas_vx0 - dust_ux0[i,:])*np.real(q_m*np.conjugate(gas_vx)) + \
               4*(gas_vy0 - dust_uy0[i,:])*np.real(q_m*np.conjugate(gas_vy)) + \
               np.abs(gas_vx - dust_ux)*np.abs(gas_vx - dust_ux) + \
               4*np.abs(gas_vy - dust_uy)*np.abs(gas_vy - dust_uy) + \
               np.abs(gas_vz - dust_uz)*np.abs(gas_vz - dust_uz))/stokes_numbers[i]

        u_5 += stkw*mu0[i,:]*dust_uz0[i,:]*np.real(q_m*np.conjugate(gas_vz))/stokes_numbers[i]

        u_tot += stkw*mu0[i,:]*(np.abs(dust_ux)*np.abs(dust_ux) + \
                4*np.abs(dust_uy)*np.abs(dust_uy) + \
                np.abs(dust_uz)*np.abs(dust_uz))

    return u_1/np.imag(eig), u_2/np.imag(eig), u_3/np.imag(eig), \
      u_4/np.imag(eig), u_5/np.imag(eig), u_tot
