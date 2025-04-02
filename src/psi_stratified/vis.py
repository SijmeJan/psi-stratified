import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .energy import energy_decomposition
from .tools import norm_factor_dust_density
from .tracker import TrackerFile
from .strat_mode import StratBox

def plot_equilibrium(sb, fig=None, color_start=0, linestyle='-', label='',
                     reference_stokes=None):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.rcParams.update({'axes.labelsize': 'large'})
    plt.rc('font', family='serif')

    if reference_stokes is None:
        zmax = \
          0.05*np.sqrt(sb.param['viscous_alpha']/1.0e-6)*\
          np.sqrt(0.01/sb.equilibrium.stokes_numbers[-1])
    else:
        zmax = \
          0.05*np.sqrt(sb.param['viscous_alpha']/1.0e-6)*\
          np.sqrt(0.01/reference_stokes)

    z = np.linspace(-zmax, zmax, 1000)

    rhog, sigma, mu, dust_rho, vx, vy, vz, ux, uy, uz = \
        sb.equilibrium.get_state(z)

    if fig is None:
        fig, ax = plt.subplots(5, 1, sharex=True, figsize=[6.4,10.4])
    else:
        ax = fig.axes

    #plt.suptitle('Equilibrium solution')

    ax[0].set_ylabel(r'$\mu$')
    ax[0].plot(z, dust_rho, linewidth=2,
               linestyle=linestyle,
               color=colors[color_start],
               label=label)
    #for i in range(0, len(uz)):
    #    ax[0].plot(z, mu[i,:], linestyle=linestyle, color=colors[(color_start+i+1) % len(colors)])

    #ax[0].set_yscale('log')
    #ax[0].set_ylim([1.0e-3,None])

    ax[1].set_ylabel(r'$\rho_{\rm g}$')
    ax[1].plot(z, rhog, linestyle=linestyle, color=colors[color_start])

    ax[2].set_ylabel(r'$u_z$')
    for i in range(0, len(uz)):
        ax[2].plot(z, uz[i,:], linestyle=linestyle, color=colors[(color_start+i+1) % len(colors)])

    ax[3].set_ylabel(r'$u_x, v_x$')
    for i in range(0, len(uz)):
        ax[3].plot(z, ux[i,:], linestyle=linestyle, color=colors[(color_start+i+1) % len(colors)])
    ax[3].plot(z, vx, linestyle=linestyle, color=colors[color_start])

    ax[4].set_ylabel(r'$u_y, v_y$')
    for i in range(0, len(uz)):
        ax[4].plot(z, uy[i,:],
                   linestyle=linestyle,
                   color=colors[(color_start+1+i) % len(colors)],
                   label=r'St={:.4f}'.format(sb.equilibrium.stokes_numbers[i]))
    ax[4].plot(z, vy, linestyle=linestyle, color=colors[color_start], label='gas')

    ax[4].legend()

    ax[4].set_xlabel(r'$z/H$')
    plt.tight_layout()

    return fig

def plot_eigenmode(sb, label):  #eig, u, z, n_dust=1):
    kx, eig, vec, z, u, du = sb.read_mode_from_file(label)

    n_eq = 4 + 4*sb.param['n_dust']

    fig, axes = plt.subplots(5, 1, sharex=True)

    plt.suptitle('Eigenvalue: '+ str(eig))

    # Normalize by largest dust density perturbation
    norm_fac = norm_factor_dust_density(u, sb.equilibrium.eq_sigma.evaluate(z),
                                        sb.equilibrium.stokes_numbers, sb.equilibrium.weights)
    u = u/norm_fac

    # Calculate total dust density perturbation
    dust_rho = sb.equilibrium.eq_sigma.evaluate(z)[0,:]*u[4,:]*sb.equilibrium.stokes_numbers[0]*sb.equilibrium.weights[0]
    dust_rho0 = sb.equilibrium.eq_sigma.evaluate(z)[0,:]*sb.equilibrium.stokes_numbers[0]*sb.equilibrium.weights[0]
    for i in range(1, len(sb.equilibrium.stokes_numbers)):
        dust_rho = dust_rho + \
          sb.equilibrium.eq_sigma.evaluate(z)[i,:]*u[4*(i+1),:]*sb.equilibrium.stokes_numbers[i]*sb.equilibrium.weights[i]
        dust_rho0 = dust_rho0 + \
          sb.equilibrium.eq_sigma.evaluate(z)[i,:]*sb.equilibrium.stokes_numbers[i]*sb.equilibrium.weights[i]
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
        elif n == 1:
            axes[n].plot(z, np.real(u[i,:]))
            axes[n].plot(z, np.imag(u[i,:]))
        else:
            axes[n].plot(z, np.abs(u[i,:]))

    return fig

def plot_coefficients(sb, label):
    kx, eig, vec, z, u, du = sb.read_mode_from_file(label)

    n_eq = 4 + 4*sb.param['n_dust']

    fig, ax = plt.subplots(1, 1)

    plt.suptitle('Eigenvalue: '+ str(eig))

    vec = vec.reshape((n_eq, int(len(vec)/n_eq)))

    ax.set_yscale('log')
    ax.set_ylabel(r'$v$')
    ax.set_xlabel(r'$j$')

    for i in range(0, n_eq):
        ax.plot(np.abs(vec[i,:]))

    return fig

def plot_energy_decomposition(sb, label): #eig, vec, kx, u, du, z):
    kx, eig, vec, z, u, du = sb.read_mode_from_file(label)

    # omega = i*s + w
    fig, ax = plt.subplots(1, 1)

    U1, U2, U3, U4, U5, Utot = energy_decomposition(z, sb, eig, kx, u, du)

    plt.suptitle('Eigenvalue: '+ str(eig))

    ax.plot(z, U1, label='vert. shear')
    ax.plot(z, U2, label='settling')
    ax.plot(z, U3, label='pressure')
    ax.plot(z, U4, label='drift')
    ax.plot(z, U5, label='buoyancy')
    ax.plot(z, Utot, label='total')

    ax.legend()
    return fig

def plot_contour_dust_density(sb, eig, vec, z):
    kx, eig, vec, z, u, du = sb.read_mode_from_file(label)

    n_eq = 4 + 4*sb.param['n_dust']

    fig, axes = plt.subplots(2, 1, sharex=True)

    plt.suptitle('Eigenvalue: '+ str(eig))
    #u = sb.evaluate_velocity_form(z, vec)

    # Normalize by largest dust density perturbation
    norm_fac = norm_factor_dust_density(u, sb.equilibrium.sigma(z),
                                        sb.equilibrium.stokes_numbers, sb.equilibrium.weights)
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

    axes[0].contourf(z, sb.equilibrium.stokes_numbers, np.real(sigma))
    axes[1].contourf(z, sb.equilibrium.stokes_numbers, np.imag(sigma))

    return fig

def plot_stokes_dist(eq):
    stokes_numbers = eq.stokes_numbers
    sigma = eq.stokes_density.sigma(stokes_numbers)

    fig, ax = plt.subplots(1, 1)

    plt.plot(stokes_numbers, sigma)

    plt.xlabel(r'$\mathrm{St}$')
    plt.ylabel(r'$\sigma_0$')
    plt.xscale('log')
    plt.yscale('log')

    return fig

def plot_pdf(filename, label):
    sb = StratBox.from_file(filename)

    pp = PdfPages('temp.pdf')

    fig = plot_equilibrium(sb)
    pp.savefig(fig)
    plt.close(fig)

    if sb.param['n_dust'] > 1:
        fig = plot_stokes_dist(sb.equilibrium)
        pp.savefig(fig)
        plt.close(fig)

    for l in sb.get_modes_in_label(label):
        fig = plot_eigenmode(sb, label + '/' + l)
        pp.savefig(fig)
        plt.close(fig)

        if sb.param['n_dust'] > 1:
            fig = plot_contour_dust_density(sb, label + '/' + l)
            pp.savefig(fig)
            plt.close(fig)

        fig = plot_energy_decomposition(sb, label + '/' + l)
        pp.savefig(fig)
        plt.close(fig)

        fig = plot_coefficients(sb, label + '/' + l)
        pp.savefig(fig)
        plt.close(fig)

    pp.close()

def plot_wavenumber_range(filename):
    fig, ax = plt.subplots(1, 1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$k_xH$')
    ax.set_ylabel(r'$\Im(\omega/\Omega)$')

    tf = TrackerFile(filename)

    #tf.delete_group('SI')
    #tf_add = TrackerFile('/Users/sjp/python/linB_tracks.h5')
    #tf.add_from_file(tf_add)

    for group_name in tf.list_groups():
        x, y, N, L = tf.read_group(group_name)

        if y.ndim == 1:
            f = np.imag(y)
        else:
            f = np.imag(y[0,:])
        ax.plot(x, f, label=group_name)

    ax.legend()

    return fig
