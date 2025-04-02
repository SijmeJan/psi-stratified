import numpy as np
import matplotlib.pyplot as plt

from equilibrium import Equilibrium
from stokesdensity import StokesDensity

# Parameters reproducing Lin (2021) figure 1
stokes = 0.01
metallicity = 0.03
viscous_alpha = 1.0e-6
n_coll = 100              # Number of BVP collocation points

# Monodisperse
stokes_density = StokesDensity(stokes)

equilibrium = Equilibrium(metallicity, stokes_density, 1, viscous_alpha)
equilibrium.solve_horizontal_velocities(n_coll, neglect_gas_viscosity=True)

z = np.linspace(0, 0.05, 100)
rhog, sigma, dg_ratio, dust_rho, gas_vx, gas_vy, gas_vz, dust_ux, dust_uy, dust_uz = equilibrium.get_state(z, eta=0.05)

fig, axs = plt.subplots(2,2)

axs[0,0].set_xlabel(r'$z$')
axs[0,0].set_ylabel(r'$\rho_{\rm g},\rho_{\rm d}$')
axs[0,0].plot(z, rhog)
axs[0,0].plot(z, dust_rho)

axs[0,1].set_xlabel(r'$z$')
axs[0,1].set_ylabel(r'$v_x,u_x$')
axs[0,1].plot(z, gas_vx)
axs[0,1].plot(z, dust_ux[0,:])

axs[1,0].set_xlabel(r'$z$')
axs[1,0].set_ylabel(r'$v_y,u_y$')
axs[1,0].plot(z, gas_vy)
axs[1,0].plot(z, dust_uy[0,:])

axs[1,1].set_xlabel(r'$z$')
axs[1,1].set_ylabel(r'$v_z,u_z$')
axs[1,1].plot(z, gas_vz)
axs[1,1].plot(z, dust_uz[0,:])

plt.tight_layout()

plt.show()

