import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cmcrameri.cm as cm
from multiprocessing import Pool, cpu_count
from atmosentry.meteoroid import Meteoroid
from atmosentry import Simulation

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def set_size(width, fraction=1, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    This code is from: https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
        width_pt = width_pt / 72.27

    elif width == 'beamer':
        width_pt = 307.28987
        width_pt = width_pt / 72.27

    else:
        width_pt = width

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    fig_width_in = width_pt * fraction
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def compute_impact(params):

    rho_com, theta0, rad, v0 = params
    M0 = rho_com * (4 * np.pi / 3) * (rad ** 3)

    impactor = Meteoroid(
        x=0,
        y=0,
        z=100e3,
        vx=-v0 * np.cos(theta0),
        vy=0,
        vz=-v0 * np.sin(theta0),
        radius=rad,
        mass=M0,
        sigma=1e4,
        rho=rho_com,
        eta=2.5e6
    )

    sim = Simulation()

    sim.impactor = impactor
    sim.dt = 1e-1

    sim.integrate()

    if len(sim.fragments):
        fragments_surface = [fragment for fragment in sim.fragments if fragment.state.z[-1] < 1]
        v_frags = np.array([
            np.sqrt(fragment.state.vx[-1] ** 2 +
                    fragment.state.vy[-1] ** 2 +
                    fragment.state.vz[-1] ** 2)
            for fragment in fragments_surface
        ])
        masses = np.array([fragment.state.mass[-1] for fragment in fragments_surface])

        vel = np.sum(v_frags * masses) / np.sum(masses)
        m = np.sum(masses)
    else:
        vel = np.sqrt(sim.impactor.state.vx ** 2 +
                      sim.impactor.state.vy ** 2 +
                      sim.impactor.state.vz ** 2)[-1]
        m = sim.impactor.state.mass[-1]

    return vel / v0, m / M0


def main():

    fig_width, fig_height = set_size('thesis')

    rho_com = 0.6e3
    theta0 = 45. * np.pi / 180.

    radii = np.logspace(1, 4, 100)
    vels = np.linspace(11.19, 30, 100) * 1e3

    r_grid, v_grid = np.meshgrid(radii, vels)

    params = [(rho_com, theta0, rad, v0) for rad in radii for v0 in vels]

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(compute_impact, params)

    v_imp = np.zeros_like(r_grid)
    m_imp = np.zeros_like(r_grid)

    for idx, (vel_ratio, mass_ratio) in enumerate(results):
        
        i = idx // len(vels)
        j = idx % len(vels)

        v_imp[j, i] = vel_ratio
        m_imp[j, i] = mass_ratio

    fig, axs = plt.subplots(1, 2, figsize=(1.25 * fig_width, fig_height), constrained_layout=True)

    sc1 = axs[0].scatter(r_grid / 1e3, v_grid / 1e3, c=v_imp, cmap=cm.oslo_r, norm=matplotlib.colors.LogNorm(vmin=1e-3, vmax=1), rasterized=True)

    axs[0].set_xscale('log')

    axs[0].set_xlim(1e-2, 1e1)
    axs[0].set_ylim(11.19, 30)

    axs[0].set_xticks([1e-2, 1e-1, 1e0, 1e1])
    axs[0].set_xticklabels([0.01, 0.1, 1.0, 10])

    axs[0].set_xlabel('Initial radius [km]', fontsize=13)
    axs[0].set_ylabel('Initial velocity [km/s]', fontsize=13)

    cbar1 = fig.colorbar(sc1, ax=axs[0], orientation='horizontal', pad=0.05, location='top', norm=matplotlib.colors.LogNorm(vmin=1e-3, vmax=1), extend='min')
    cbar1.set_label(r'$v_{\rm surf}/v_{\rm init}$', fontsize=13)
    # cbar1.ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    sc2 = axs[1].scatter(r_grid / 1e3, v_grid / 1e3, c=m_imp, cmap=cm.oslo_r, norm=matplotlib.colors.LogNorm(vmin=1e-2, vmax=1), rasterized=True)

    axs[1].set_xscale('log')

    axs[1].set_xlim(1e-2, 1e1)
    axs[1].set_ylim(11.19, 30)

    axs[1].set_xticks([1e-2, 1e-1, 1e0, 1e1])
    axs[1].set_xticklabels([0.01, 0.1, 1.0, 10])

    axs[1].set_yticks([15, 20, 25, 30])
    axs[1].set_yticklabels([])

    axs[1].set_xlabel('Initial radius [km]', fontsize=13)

    cbar2 = fig.colorbar(sc2, ax=axs[1], orientation='horizontal', pad=0.05, location='top', norm=matplotlib.colors.LogNorm(vmin=1e-2, vmax=1), extend='min')
    cbar2.set_label(r'$m_{\rm surf}/m_{\rm init}$', fontsize=13)
    # cbar2.ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    axs[0].minorticks_on()
    axs[1].minorticks_on()

    plt.savefig('./paper/figures/heatmaps.pdf', bbox_inches='tight', format='pdf')

    plt.show()

    results = {
        'r_grid': r_grid,
        'v_grid': v_grid,
        'v_imp': v_imp,
        'm_imp': m_imp
    }
    np.savez('./paper/figures/heatmap_data.npz', **results)


if __name__ == "__main__":

    # main()

    fig_width, fig_height = set_size('thesis')

    data = np.load('./paper/figures/heatmap_data.npz')
    v_imp = data['v_imp']
    m_imp = data['m_imp']
    v_grid = data['v_grid']
    r_grid = data['r_grid']

    fig, axs = plt.subplots(1, 2, figsize=(1.25 * fig_width, fig_height), constrained_layout=True)

    sc1 = axs[0].scatter(r_grid / 1e3, v_grid / 1e3, c=v_imp, cmap='coolwarm', norm=matplotlib.colors.LogNorm(vmin=1e-3, vmax=1), rasterized=True)
    # cmaps = [PuBu, coolwarm, Spectral] (in order)

    axs[0].set_xscale('log')

    axs[0].set_xlim(1e-2, 1e0)
    axs[0].set_ylim(11.19, 30)

    axs[0].set_xticks([1e-2, 1e-1, 1e0])
    axs[0].set_xticklabels([0.01, 0.1, 1.0])

    axs[0].set_xlabel('Initial radius [km]', fontsize=13)
    axs[0].set_ylabel('Initial velocity [km/s]', fontsize=13)

    cbar1 = fig.colorbar(sc1, ax=axs[0], orientation='horizontal', pad=0.05, location='top', norm=matplotlib.colors.LogNorm(vmin=1e-3, vmax=1), extend='min')
    cbar1.set_label(r'$v_{\rm surf}/v_{\rm init}$', fontsize=13)

    sc2 = axs[1].scatter(r_grid / 1e3, v_grid / 1e3, c=m_imp, cmap='coolwarm', norm=matplotlib.colors.LogNorm(vmin=1e-2, vmax=1), rasterized=True)

    axs[1].set_xscale('log')

    axs[1].set_xlim(1e-2, 1e0)
    axs[1].set_ylim(11.19, 30)

    axs[1].set_xticks([1e-2, 1e-1, 1e0])
    axs[1].set_xticklabels([0.01, 0.1, 1.0])

    axs[1].set_yticks([15, 20, 25, 30])
    axs[1].set_yticklabels([])

    axs[1].set_xlabel('Initial radius [km]', fontsize=13)

    cbar2 = fig.colorbar(sc2, ax=axs[1], orientation='horizontal', pad=0.05, location='top', norm=matplotlib.colors.LogNorm(vmin=1e-2, vmax=1), extend='min')
    cbar2.set_label(r'$m_{\rm surf}/m_{\rm init}$', fontsize=13)

    axs[0].minorticks_on()
    axs[1].minorticks_on()

    # plt.savefig('./paper/figures/heatmaps.pdf', bbox_inches='tight', format='pdf')

    plt.show()