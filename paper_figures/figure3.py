# pylint: disable=C0103,W0621,C0200,E1101

"""
Script to generate figure 3 from Anslow+ 2025 (MNRAS, subm.)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from cmcrameri import cm
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


def generate_axes(fig):
    """
    Generate axes for multi-panel figure
    """

    gridspec = fig.add_gridspec(nrows=6, ncols=12, height_ratios=[3, 1, 1, 1, 1, 1])

    axes = {}
    axes['1'] = fig.add_subplot(gridspec[0:2, 4:8])
    axes['2'] = fig.add_subplot(gridspec[2:4, 3:6])
    axes['3'] = fig.add_subplot(gridspec[2:4, 6:9])
    axes['4'] = fig.add_subplot(gridspec[4:6, 3:6])
    axes['5'] = fig.add_subplot(gridspec[4:6, 6:9])

    return axes

fig_width, fig_height = set_size('thesis', 1, (1, 1))

rho_com = 0.6e3
rho_atm0 = 1.225

theta0 = 45. * np.pi / 180.
V0 = 20e3
R0 = [10, 50, 150, 500, 1000]

fig = plt.figure(figsize=(2.5 * fig_width, 2 * fig_height))

axes = generate_axes(fig)

ax1 = axes['1']
ax2 = axes['2']
ax3 = axes['3']
ax4 = axes['4']
ax5 = axes['5']

for i in range(len(R0)):

    M0 = rho_com * (4 * np.pi / 3) * (R0[i] ** 3)
    KE0 = 0.5 * M0 * (V0 ** 2)

    impactor = Meteoroid(x=0,
                        y=0,
                        z=100e3,
                        vx=-V0 * np.cos(theta0),
                        vy=0,
                        vz=-V0 * np.sin(theta0),
                        radius=R0[i],
                        mass=M0,
                        sigma=1e4,
                        rho=0.6e3,
                        eta=2.5e6)

    sim = Simulation()

    sim.rho0 = rho_atm0

    sim.impactor = impactor

    sim.integrate()

    vel = np.sqrt(sim.impactor.state.vx ** 2 +
                  sim.impactor.state.vy ** 2 +
                  sim.impactor.state.vz ** 2)

    ax1.plot(vel / 1e3, sim.impactor.state.z / 1e3, c=cm.bamako((len(R0) - i) / len(R0)),
             label=fr'$R_0=$ {R0[i]} m')

    if len(sim.fragments):
        ax1.plot(vel[-1] / 1e3, sim.impactor.state.z[-1] / 1e3, 'x', c='k')

        for fragment in sim.fragments:

            vel = np.sqrt(fragment.state.vx ** 2 + fragment.state.vy ** 2 + fragment.state.vz ** 2)

            ax1.plot(vel / 1e3, fragment.state.z / 1e3, c=cm.bamako((len(R0) - i) / len(R0)), )
            if fragment.state.z[-1] > 1:
                if fragment.children:

                    ax1.plot(vel[-1] / 1e3, fragment.state.z[-1] / 1e3, 'x', c='k', alpha=0.5)

    if len(sim.fragments):
        fragments_surface = [fragment for fragment in sim.fragments if fragment.state.z[-1] < 1]
        masses = [fragment.state.mass[-1] / M0 for fragment in fragments_surface]
        vels = [np.sqrt(fragment.state.vx[-1] ** 2 +
                        fragment.state.vy[-1] ** 2 +
                        fragment.state.vz[-1] ** 2) / 1e3 for fragment in fragments_surface]

        ax2.scatter(masses, vels, marker='.', color=cm.bamako((len(R0) - i) / len(R0)),
                    label=rf'$R_0=$ {R0[i]} m')
    else:
        vel = np.sqrt(sim.impactor.state.vx ** 2 +
                      sim.impactor.state.vy ** 2 +
                      sim.impactor.state.vz ** 2)

        ax2.scatter(sim.impactor.state.mass[-1] / M0, vel[-1] / 1e3,
                    color=cm.bamako((len(R0) - i) / len(R0)),
                    marker='.', label=rf'$R_0=$ {R0[i]} m')

    if len(sim.fragments):
        fragments_surface = [fragment for fragment in sim.fragments if fragment.state.z[-1] < 1]
        rads = [fragment.state.radius[-1] / R0[i] for fragment in fragments_surface]
        vels = [np.sqrt(fragment.state.vx[-1] ** 2 +
                        fragment.state.vy[-1] ** 2 +
                        fragment.state.vz[-1] ** 2) / 1e3 for fragment in fragments_surface]

        ax3.scatter(rads, vels, marker='.', color=cm.bamako((len(R0) - i) / len(R0)))
    else:
        vel = np.sqrt(sim.impactor.state.vx ** 2 +
                      sim.impactor.state.vy ** 2 +
                      sim.impactor.state.vz ** 2)

        ax3.scatter(sim.impactor.state.radius[-1] / R0[i], vel[-1] / 1e3,
                    color=cm.bamako((len(R0) - i) / len(R0)), marker='.')

    altitudes = np.linspace(0, 100e3, 1000)
    cumulative_energy_deposition = np.zeros_like(altitudes)
    cumulative_mass_deposition = np.zeros_like(altitudes)

    for j, h in enumerate(altitudes):
        cumulative_dE = 0.0
        cumulative_dM = 0.0

        cumulative_dE += np.sum(sim.impactor.state.dEkin[sim.impactor.state.z >= h])
        cumulative_dM += np.sum(sim.impactor.state.dM[sim.impactor.state.z >= h])

        for fragment in sim.fragments:
            cumulative_dE += np.sum(fragment.state.dEkin[fragment.state.z >= h])
            cumulative_dM += np.sum(fragment.state.dM[fragment.state.z >= h])

        cumulative_energy_deposition[j] = cumulative_dE
        cumulative_mass_deposition[j] = cumulative_dM

    ax4.plot(cumulative_mass_deposition / M0, altitudes / 1e3,
             c=cm.bamako((len(R0) - i) / len(R0)), label=rf'$R_0=$ {R0[i]} m')
    ax5.plot(cumulative_energy_deposition / KE0, altitudes / 1e3,
             c=cm.bamako((len(R0) - i) / len(R0)), label=rf'$R_0=$ {R0[i]} m')

ax1.set_yscale('log')
ax1.set_ylim(1e-1, 100)
ax1.set_yticks([0.1, 1, 10, 100], labels=[0.1, 1, 10, 100])
ax1.set_xlabel(r'Velocity [km/s]', fontsize=13)
ax1.set_ylabel(r'Altitude [km]', fontsize=13)
ax1.minorticks_on()
ax1.legend(frameon=False, loc=(0.1, 0.03))

ax2.axhline(20, c='tab:gray', ls='--', zorder=0, alpha=0.5)
ax2.axvline(1, c='tab:gray', ls='--', zorder=0, alpha=0.5)
ax2.set_xlabel(r'Fragment mass $[m_{\rm frag}/M_0]$', fontsize=13)
ax2.set_ylabel('Velocity [km/s]', fontsize=13)
ax2.minorticks_on()

ax3.axhline(20, c='tab:gray', ls='--', zorder=0, alpha=0.5)
ax3.axvline(1, c='tab:gray', ls='--', zorder=0, alpha=0.5)
ax3.set_xlabel(r'Fragment radius $[r_{\rm frag}/R_0]$', fontsize=13)
ax3.set_ylabel('Velocity [km/s]', fontsize=13)
ax3.minorticks_on()

ax4.set_xscale('log')
ax4.set_xlim(1e-3, 2)
ax4.set_yscale('log')
ax4.set_ylim(1e-1, 100)
ax4.set_yticks([0.1, 1, 10, 100], labels=[0.1, 1, 10, 100])
ax4.set_xlabel(r'Cumulative mass loss [$M_0$]', fontsize=13)
ax4.set_ylabel('Altitude [km]', fontsize=13)
ax4.minorticks_on()

ax5.set_xscale('log')
ax5.set_xlim(1e-3, 2)
ax5.set_yscale('log')
ax5.set_ylim(1e-1, 100)
ax5.set_yticks([0.1, 1, 10, 100], labels=[0.1, 1, 10, 100])
ax5.set_xlabel(r'Cumulative energy deposition [$E_0$]', fontsize=13)
ax5.set_ylabel('Altitude [km]', fontsize=13)
ax5.minorticks_on()

fig.tight_layout()

ttt = ['a', 'b', 'c', 'd', 'e']
axs = [ax1, ax2, ax3, ax4, ax5]
for p, l in zip(axs, ttt):
    p.annotate(l, xy=(-0., 1.04), xycoords="axes fraction", fontsize=10, weight='bold')

with PdfPages('./paper_figures/figures/comet_trajectory_gallery.pdf') as pdf:
    pdf.savefig(fig, bbox_inches='tight', transparent=True)

plt.show()
