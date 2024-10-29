import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import cmcrameri.cm as cm
from atmosentry.meteoroid import Meteoroid
from atmosentry import Simulation
from chyba_model import run_intergration_chyba

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

fig6, ax6 = plt.subplots(1, 1, figsize=(fig_width, fig_height))

for i in range(len(R0)):

    M0 = rho_com * (4 * np.pi / 3) * (R0[i] ** 3)

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

    sim.impactor = impactor

    sim.integrate()

    vel = np.sqrt(sim.impactor.vx ** 2 + sim.impactor.vy ** 2 + sim.impactor.vz ** 2)

    ax1.plot(vel / 1e3, sim.impactor.z / 1e3, c=cm.bamako((len(R0) - i) / len(R0)), label=fr'$R_0=$ {R0[i]} m')
    ax6.plot(vel / 1e3, sim.impactor.z / 1e3, c=cm.bamako((len(R0) - i) / len(R0)), label=fr'$R_0=$ {R0[i]} m')
    if len(sim.fragments):
        ax1.plot(vel[-1] / 1e3, sim.impactor.z[-1] / 1e3, 'x', c='k')
        ax6.plot(vel[-1] / 1e3, sim.impactor.z[-1] / 1e3, 'x', c='k', alpha=0.5)

        for fragment in sim.fragments:

            vel = np.sqrt(fragment.vx ** 2 + fragment.vy ** 2 + fragment.vz ** 2)

            ax1.plot(vel / 1e3, fragment.z / 1e3, c=cm.bamako((len(R0) - i) / len(R0)), )
            ax6.plot(vel / 1e3, fragment.z / 1e3, c=cm.bamako((len(R0) - i) / len(R0)), alpha=0.5)
            if fragment.z[-1] > 1:
                if fragment.children:

                    ax1.plot(vel[-1] / 1e3, fragment.z[-1] / 1e3, 'x', c='k', alpha=0.5)
                    ax6.plot(vel[-1] / 1e3, fragment.z[-1] / 1e3, 'x', c='k', alpha=0.5)

    _, vel_chyba, mass_chyba, _, altitude_chyba, _, _, _ =\
            run_intergration_chyba(V0, M0, theta0, 100e3, R0[i], 0, 1e4, rho_com, 2.5e6)
    ax6.plot(vel_chyba / 1e3, altitude_chyba / 1e3, color=cm.bamako((len(R0) - i - 0.5) / len(R0)), ls='--')

    if len(sim.fragments):
        fragments_surface = [fragment for fragment in sim.fragments if fragment.z[-1] < 1]
        masses = [fragment.mass[-1] / M0 for fragment in fragments_surface]
        vels = [np.sqrt(fragment.vx[-1] ** 2 + fragment.vy[-1] ** 2 + fragment.vz[-1] ** 2) / 1e3 for fragment in fragments_surface]

        ax2.scatter(masses, vels, marker='.', color=cm.bamako((len(R0) - i) / len(R0)), label=rf'$R_0=$ {R0[i]} m')
    else:
        vel = np.sqrt(sim.impactor.vx ** 2 + sim.impactor.vy ** 2 + sim.impactor.vz ** 2)
        ax2.scatter(sim.impactor.mass[-1] / M0, vel[-1] / 1e3, color=cm.bamako((len(R0) - i) / len(R0)), marker='.', label=rf'$R_0=$ {R0[i]} m')

    if len(sim.fragments):
        fragments_surface = [fragment for fragment in sim.fragments if fragment.z[-1] < 1]
        rads = [fragment.radius[-1] / R0[i] for fragment in fragments_surface]
        vels = [np.sqrt(fragment.vx[-1] ** 2 + fragment.vy[-1] ** 2 + fragment.vz[-1] ** 2) / 1e3 for fragment in fragments_surface]

        ax3.scatter(rads, vels, marker='.', color=cm.bamako((len(R0) - i) / len(R0)))
    else:
        vel = np.sqrt(sim.impactor.vx ** 2 + sim.impactor.vy ** 2 + sim.impactor.vz ** 2)
        ax3.scatter(sim.impactor.radius[-1] / R0[i], vel[-1] / 1e3, color=cm.bamako((len(R0) - i) / len(R0)), marker='.')

    # ax3.plot(impactor.radius / R0[i], impactor.z / 1e3, color=cm.bamako((len(R0) - i) / len(R0)))

    # if len(sim.fragments):
    #     ax3.plot(impactor.radius[-1] / R0[i], impactor.z[-1] / 1e3, 'x', color='k', alpha=0.5)

    #     for fragment in sim.fragments:

    #         ax3.plot(fragment.radius / R0[i], fragment.z / 1e3, color=cm.bamako((len(R0) - i) / len(R0)))
    #         if fragment.z[-1] > 1:
    #             ax3.plot(fragment.radius[-1] / R0[i], fragment.z[-1] / 1e3, 'x', color='k', alpha=0.5)

    altitudes = np.linspace(0, 100e3, 1000)
    cumulative_energy_deposition = np.zeros_like(altitudes)
    cumulative_mass_deposition = np.zeros_like(altitudes)

    for j, h in enumerate(altitudes):
        cumulative_dE = 0.0
        cumulative_dM = 0.0

        cumulative_dE += np.sum(sim.impactor.dEkin[sim.impactor.z >= h])
        cumulative_dM += np.sum(sim.impactor.dM[sim.impactor.z >= h])

        for fragment in sim.fragments:
            cumulative_dE += np.sum(fragment.dEkin[fragment.z >= h])
            cumulative_dM += np.sum(fragment.dM[fragment.z >= h])
        
        cumulative_energy_deposition[j] = cumulative_dE
        cumulative_mass_deposition[j] = cumulative_dM

    ax4.plot(cumulative_mass_deposition / M0, altitudes / 1e3, c=cm.bamako((len(R0) - i) / len(R0)), label=rf'$R_0=$ {R0[i]} m')
    ax5.plot(cumulative_energy_deposition, altitudes / 1e3, c=cm.bamako((len(R0) - i) / len(R0)), label=rf'$R_0=$ {R0[i]} m')

ax1.set_ylim(0, 60)
ax1.set_xlabel(r'Velocity [km/s]', fontsize=13)
ax1.set_ylabel(r'Altitude [km]', fontsize=13)
ax1.minorticks_on()
ax1.legend(frameon=False, loc='upper left')

ax2.axhline(20, c='tab:gray', ls='--', zorder=0, alpha=0.5)
ax2.axvline(1, c='tab:gray', ls='--', zorder=0, alpha=0.5)
ax2.set_xlabel(r'Fragment mass $[m_{\rm frag}/M_0]$', fontsize=13)
ax2.set_ylabel('Velocity [km/s]', fontsize=13)
ax2.minorticks_on()

# ax3.axvline(1e1, c='tab:gray', ls='--', zorder=0, alpha=0.5)
# ax3.set_xscale('log')
ax3.axhline(20, c='tab:gray', ls='--', zorder=0, alpha=0.5)
ax3.axvline(1, c='tab:gray', ls='--', zorder=0, alpha=0.5)
ax3.set_xlabel(r'Fragment radius $[r_{\rm frag}/R_0]$', fontsize=13)
ax3.set_ylabel('Altitude [km]', fontsize=13)
ax3.minorticks_on()

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 60)
ax4.set_xlabel(r'Cumulative mass loss [$M_0$]', fontsize=13)
ax4.set_ylabel('Altitude [km]', fontsize=13)
ax4.minorticks_on()

ax5.set_xscale('log')
ax5.set_xlim(1e13, 1e20)
ax5.set_ylim(0, 60)
ax5.set_xlabel(r'Cumulative energy deposition [J]', fontsize=13)
ax5.set_ylabel('Altitude [km]', fontsize=13)
ax5.minorticks_on()

ax6.minorticks_on()
ax6.set_ylim(0, 60)
ax6.set_xlabel(r'Velocity [km/s]', fontsize=13)
ax6.set_ylabel(r'Altitude [km]', fontsize=13)
ax6.legend(frameon=False, loc='upper left')


fig.tight_layout()
fig6.tight_layout()

ttt = ['a', 'b', 'c', 'd', 'e']
axs = [ax1, ax2, ax3, ax4, ax5]
for p, l in zip(axs, ttt):
    p.annotate(l, xy=(-0., 1.04), xycoords="axes fraction", fontsize=10, weight='bold')

with PdfPages('./examples/figures/comet_trajectory_gallery.pdf') as pdf:
    pdf.savefig(fig, bbox_inches='tight', )


with PdfPages('./examples/figures/chyba_comparison.pdf') as pdf:
    pdf.savefig(fig6, bbox_inches='tight', )

plt.show()
