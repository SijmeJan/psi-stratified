import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .energy import energy_decomposition
from .tools import norm_factor_dust_density

def plot_equilibrium(eq, interval=[0,1]):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)

    plt.suptitle('Equilibrium solution')
    z = np.linspace(interval[0], interval[1], 1000)

    rhog = eq.gasdens(z)
    sigma = eq.sigma(z)
    mu = eq.epsilon(z)

    dust_rho = sigma[0,:]*eq.tau[0]*eq.weights[0]
    for i in range(1, len(eq.tau)):
        dust_rho = dust_rho + sigma[i,:]*eq.tau[i]*eq.weights[i]
    dust_rho = dust_rho/rhog

    vx, vy, ux, uy = eq.evaluate(z, k=0)
    #dvx, dvy, dux, duy = eq.evaluate(z, k=1)
    #d2vx, d2vy, d2ux, d2uy = eq.evaluate(z, k=2)

    uz = eq.uz(z)

    ax1.set_ylabel(r'$\mu$')
    #ax1.set_yscale('log')
    #for i in range(0, len(uz)):
    #    ax1.plot(z, mu[i,:])
    ax1.plot(z, dust_rho, linewidth=2)

    ax2.set_ylabel(r'$\rho_{\rm g}$')
    ax2.plot(z, rhog)
    #ax2.plot(z, np.exp(-0.5*z*z))

    ax3.set_ylabel(r'$u_z$')
    for i in range(0, len(uz)):
        ax3.plot(z, uz[i,:])

    ax4.set_ylabel(r'$u_x, v_x$')
    for i in range(0, len(uz)):
        ax4.plot(z, ux[i,:])
    ax4.plot(z, vx)

    ax5.set_ylabel(r'$u_y, v_y$')
    for i in range(0, len(uz)):
        ax5.plot(z, uy[i,:])
    ax5.plot(z, vy)

    ax5.set_xlabel(r'$z/H$')
    plt.tight_layout()

    return fig

def plot_eigenmode(sb, eig, vec, interval=[0,1]):
    n_eq = 4 + 4*sb.n_dust

    fig, axes = plt.subplots(5, 1, sharex=True)

    z = np.linspace(interval[0], interval[1], 1000)

    plt.suptitle('Eigenvalue: '+ str(eig))
    u = sb.evaluate_velocity_form(z, vec)

    # Normalize by largest dust density perturbation
    norm_fac = norm_factor_dust_density(u, sb.eq.sigma(z),
                                        sb.eq.tau, sb.eq.weights)
    u = u/norm_fac

    # Calculate total dust density perturbation
    dust_rho = sb.eq.sigma(z)[0,:]*u[4,:]*sb.eq.tau[0]*sb.eq.weights[0]
    dust_rho0 = sb.eq.sigma(z)[0,:]*sb.eq.tau[0]*sb.eq.weights[0]
    for i in range(1, len(sb.eq.tau)):
        dust_rho = dust_rho + \
          sb.eq.sigma(z)[i,:]*u[4*(i+1),:]*sb.eq.tau[i]*sb.eq.weights[i]
        dust_rho0 = dust_rho0 + \
          sb.eq.sigma(z)[i,:]*sb.eq.tau[i]*sb.eq.weights[i]
    dust_rho = dust_rho/dust_rho0

    axes[0].set_ylabel(r'$\sigma$')
    axes[1].set_ylabel(r'$\rho_\mathrm{g}$')
    axes[2].set_ylabel(r'$v_x$')
    axes[3].set_ylabel(r'$v_y$')
    axes[4].set_ylabel(r'$v_z$')

    axes[-1].set_xlabel(r'$z\Omega/c$')

    for i in range(0, n_eq):
        n = i % 4
        if i == 0:
            n = 1     # Gas density in second panel
        else:
            if n > 0:      # Shift everything down except dust density
                n = n + 1

        if n == 0:
            if i == 4:
                axes[n].plot(z, np.real(dust_rho), linewidth=2)
                axes[n].plot(z, np.imag(dust_rho), linewidth=2)

            #print(i)
            #axes[n].plot(z, np.real(u[i,:]))
            #axes[n].plot(z, np.imag(u[i,:]))
        elif n == 1:
            axes[n].plot(z, np.real(u[i,:]))
            axes[n].plot(z, np.imag(u[i,:]))
        else:
            axes[n].plot(z, np.abs(u[i,:]))

    #plt.show()
    return fig

def plot_coefficients(sb, eig, vec):
    n_eq = 4 + 4*sb.n_dust

    fig, ax = plt.subplots(1, 1)

    plt.suptitle('Eigenvalue: '+ str(eig))

    vec = vec.reshape((n_eq, int(len(vec)/n_eq)))

    ax.set_yscale('log')
    ax.set_ylabel(r'$v$')
    ax.set_xlabel(r'$j$')

    for i in range(0, n_eq):
        ax.plot(np.abs(vec[i,:]))

    #plt.show()
    return fig

def plot_energy_decomposition(sb, eig, vec, kx, interval=[0,1]):
    # omega = i*s + w
    fig, ax = plt.subplots(1, 1)

    z = np.linspace(interval[0], interval[1], 1000)

    U1, U2, U3, U4, U5, Utot = energy_decomposition(z, sb, eig, vec, kx)

    plt.suptitle('Eigenvalue: '+ str(eig))

    ax.plot(z, U1, label='vert. shear')
    ax.plot(z, U2, label='settling')
    ax.plot(z, U3, label='pressure')
    ax.plot(z, U4, label='drift')
    ax.plot(z, U5, label='buoyancy')
    ax.plot(z, Utot, label='total')

    ax.legend()
    return fig

def plot_contour_dust_density(sb, eig, vec, interval=[0,1]):
    n_eq = 4 + 4*sb.n_dust

    fig, axes = plt.subplots(2, 1, sharex=True)

    z = np.linspace(interval[0], interval[1], 1000)

    plt.suptitle('Eigenvalue: '+ str(eig))
    u = sb.evaluate_velocity_form(z, vec)

    # Normalize by largest dust density perturbation
    norm_fac = norm_factor_dust_density(u, sb.eq.sigma(z),
                                        sb.eq.tau, sb.eq.weights)
    u = u/norm_fac

    sigma = np.ndarray((sb.n_dust, 1000), dtype=np.cdouble)
    for i in range(0, sb.n_dust):
        sigma[i,:] = u[4*i+4, :]

    axes[0].set_yscale('log')
    axes[1].set_yscale('log')

    axes[0].set_title(r'$\Re(\hat\sigma)$')
    axes[1].set_title(r'$\Im(\hat\sigma)$')

    axes[1].set_xlabel(r'$z$')
    axes[0].set_ylabel(r'$\mathrm{St}$')
    axes[1].set_ylabel(r'$\mathrm{St}$')

    axes[0].contourf(z, sb.eq.tau, np.real(sigma))
    axes[1].contourf(z, sb.eq.tau, np.imag(sigma))

    return fig

def plot_stokes_dist(eq):
    tau = eq.tau
    sigma = eq.stokes_density.sigma(tau)

    fig, ax = plt.subplots(1, 1)

    plt.plot(tau, sigma)

    plt.xlabel(r'$\mathrm{St}$')
    plt.ylabel(r'$\sigma_0$')
    plt.xscale('log')
    plt.yscale('log')

    return fig

def plot_pdf(sb):
    zmax = 0.05*np.sqrt(sb.param['viscous_alpha']/1.0e-6)*np.sqrt(0.01/sb.eq.tau[-1])
    z_interval = [-zmax, zmax]
    pp = PdfPages('temp.pdf')

    fig = plot_equilibrium(sb.eq, interval=z_interval)
    pp.savefig(fig)

    #fig = plot_stokes_dist(sb.eq)
    #pp.savefig(fig)

    for n, v in enumerate(sb.vec):
        fig = plot_eigenmode(sb, sb.eig[n], v, interval=z_interval)
        pp.savefig(fig)
        plt.close(fig)

        #fig = plot_contour_dust_density(sb, sb.eig[n],v,interval=z_interval)
        #pp.savefig(fig)
        #plt.close(fig)

        fig = plot_energy_decomposition(sb, sb.eig[n], v, sb.kx,
                                        interval=z_interval)
        pp.savefig(fig)
        plt.close(fig)

        fig = plot_coefficients(sb, sb.eig[n], v)
        pp.savefig(fig)
        plt.close(fig)

    pp.close()

def plot_wavenumber_range(filename):
    fig, ax = plt.subplots(1, 1)
    ax.set_xscale('log')
    ax.set_xlabel(r'$k_xH$')
    ax.set_ylabel(r'$\Im(\omega/\Omega)$')

    hf = h5.File(filename, 'r')

    for group_name in hf:
        g = hf[group_name]

        # Get data
        x = g.get('x_values')[()]
        y = g.get('y_values')[()]

        ax.plot(x, np.imag(y[0,:]), label=group_name)

    ax.legend()

    hf.close()

    return fig
