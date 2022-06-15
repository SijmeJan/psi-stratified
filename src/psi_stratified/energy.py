import numpy as np

from .tools import norm_factor_dust_density

def energy_decomposition(z, sb, eig, vec, kx):
    # Equilibrium structure
    tau = sb.eq.tau
    rhog0 = sb.eq.gasdens(z)
    rhod0 = sb.eq.sigma(z)
    mu0 = sb.eq.epsilon(z)
    dlogrhogdz = sb.eq.dlogrhogdz(z)
    dlogsigmadz = sb.eq.dlogsigmadz(z)
    vx0, vy0, ux0, uy0 = sb.eq.evaluate(z, k=0)
    dvx0, dvy0, dux0, duy0 = sb.eq.evaluate(z, k=1)
    d2vx0, d2vy0, d2ux0, d2uy0 = sb.eq.evaluate(z, k=2)
    uz0 = sb.eq.uz(z)
    duz0 = sb.eq.duz(z)

    # Evaluate perturbations
    u = sb.evaluate_velocity_form(z, vec)
    du = sb.evaluate_velocity_form(z, vec, k=1)

    # Normalize to maximum dust density perturbation
    norm_fac = norm_factor_dust_density(u, rhod0, tau, sb.eq.weights)
    u = u/norm_fac
    du = du/norm_fac

    rhog = u[0,:]
    vx = u[1,:]
    vy = u[2,:]
    vz = u[3,:]
    rhod = u[4,:]
    ux = u[5,:]
    uy = u[6,:]
    uz = u[7,:]

    drhog = du[0,:]
    dvx = du[1,:]
    dvy = du[2,:]
    dvz = du[3,:]
    drhod = du[4,:]
    dux = du[5,:]
    duy = du[6,:]
    duz = du[7,:]

    f = sb.eq.tau[0]*sb.eq.weights[0]
    U1 = -(f*mu0[0,:]*dux0[0,:]*np.real(uz*np.conjugate(ux)) + \
           dvx0*np.real(vz*np.conjugate(vx))) - \
          4*(f*mu0[0,:]*duy0[0,:]*np.real(uz*np.conjugate(uy)) + \
             dvy0*np.real(vz*np.conjugate(vy))) - \
          f*mu0[0,:]*duz0[0,:]*np.abs(uz)*np.abs(uz)

    U2 = -f*mu0[0,:]*uz0[0,:]*np.real(dux*np.conjugate(ux) + \
                                      4*duy*np.conjugate(uy) + \
                                      duz*np.conjugate(uz))

    W = rhog
    dW = drhog

    U3 = kx*np.imag(W*np.conjugate(vx)) - \
         np.real(dW*np.conjugate(vz))

    # Q = mu/mu0
    Q = rhod - rhog

    U4 = -f*mu0[0,:]*((vx0 - ux0[0,:])*np.real(Q*np.conjugate(vx)) + \
               4*(vy0 - uy0[0,:])*np.real(Q*np.conjugate(vy)) + \
               np.abs(vx - ux)*np.abs(vx - ux) + \
               4*np.abs(vy - uy)*np.abs(vy - uy) + \
               np.abs(vz - uz)*np.abs(vz - uz))/tau[0]


    U5 = f*mu0[0,:]*uz0[0,:]*np.real(Q*np.conjugate(vz))/tau[0]

    Utot = f*mu0[0,:]*(np.abs(ux)*np.abs(ux) + \
                4*np.abs(uy)*np.abs(uy) + \
                np.abs(uz)*np.abs(uz)) + \
      np.abs(vx)*np.abs(vx) + \
      4*np.abs(vy)*np.abs(vy) + \
      np.abs(vz)*np.abs(vz)

    for i in range(1, sb.eq.n_dust):
        rhod = u[4 + 4*i,:]
        ux = u[4 + 4*i + 1,:]
        uy = u[4 + 4*i + 2,:]
        uz = u[4 + 4*i + 3,:]
        dux = du[4 + 4*i + 1,:]
        duy = du[4 + 4*i + 2,:]
        duz = du[4 + 4*i + 3,:]

        f = sb.eq.tau[i]*sb.eq.weights[i]
        U1 += -f*mu0[i,:]*dux0[i,:]*np.real(uz*np.conjugate(ux)) - \
          4*f*mu0[i,:]*duy0[i,:]*np.real(uz*np.conjugate(uy)) - \
          f*mu0[i,:]*duz0[i,:]*np.abs(uz)*np.abs(uz)

        U2 += -f*mu0[i,:]*uz0[i,:]*np.real(dux*np.conjugate(ux) + \
                                           4*duy*np.conjugate(uy) + \
                                           duz*np.conjugate(uz))

        Q = rhod - rhog
        U4 += -f*mu0[i,:]*((vx0 - ux0[i,:])*np.real(Q*np.conjugate(vx)) + \
               4*(vy0 - uy0[i,:])*np.real(Q*np.conjugate(vy)) + \
               np.abs(vx - ux)*np.abs(vx - ux) + \
               4*np.abs(vy - uy)*np.abs(vy - uy) + \
               np.abs(vz - uz)*np.abs(vz - uz))/tau[i]

        U5 += f*mu0[i,:]*uz0[i,:]*np.real(Q*np.conjugate(vz))/tau[i]

        Utot += f*mu0[i,:]*(np.abs(ux)*np.abs(ux) + \
                4*np.abs(uy)*np.abs(uy) + \
                np.abs(uz)*np.abs(uz))

    return U1/np.imag(eig), U2/np.imag(eig), U3/np.imag(eig), \
      U4/np.imag(eig), U5/np.imag(eig), Utot
