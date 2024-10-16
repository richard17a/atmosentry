import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import cmcrameri.cm as cm
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

fig_width, fig_height = set_size('thesis')

R0 = 150
V0 = 20e3

rho_com = 0.6e3
theta0 = 45 * np.pi / 180

M0 = rho_com * (4 * np.pi / 3) * (R0 ** 3)

impactor = Meteoroid(x=0,
                        y=0,
                        z=100e3,
                        vx=-V0 * np.cos(theta0),
                        vy=0,
                        vz=-V0 * np.sin(theta0),
                        theta=theta0,
                        radius=R0,
                        mass=M0,
                        sigma=1e4,
                        rho=rho_com,
                        eta=2.5e6)

sim = Simulation()
sim.impactor = impactor
sim.integrate()

_, ax = plt.subplots(figsize=(1.25 * fig_width, fig_height))

vel = np.sqrt(sim.impactor.vx ** 2 + sim.impactor.vy ** 2 + sim.impactor.vz ** 2)

ax.plot(vel / 1e3, sim.impactor.z / 1e3, c=cm.bamako(0.6), label=fr'$R_0=$ {R0} m')

if len(sim.fragments):
    ax.plot(vel[-1] / 1e3, sim.impactor.z[-1] / 1e3, 'x', c='k', alpha=0.5)

    for fragment in sim.fragments:

        vel = np.sqrt(fragment.vx ** 2 + fragment.vy ** 2 + fragment.vz ** 2)

        ax.plot(vel / 1e3, fragment.z / 1e3, c=cm.bamako(0.6), )
        if fragment.z[-1] > 1:
            ax.plot(vel[-1] / 1e3, fragment.z[-1] / 1e3, 'x', c='k', alpha=0.5)

ax.set_ylim(0, 25)

axins = inset_axes(ax, loc='upper left', width="70%", height="40%")
if len(sim.fragments):
    axins.plot(vel[-1] / 1e3, sim.impactor.z[-1] / 1e3, 'x', c='k', alpha=0.5)

    for fragment in sim.fragments:

        vel = np.sqrt(fragment.vx ** 2 + fragment.vy ** 2 + fragment.vz ** 2)

        axins.plot(vel / 1e3, fragment.z / 1e3, c=cm.bamako(0.6), )
        if fragment.z[-1] > 1:
            axins.plot(vel[-1] / 1e3, fragment.z[-1] / 1e3, 'x', c='k', alpha=0.5)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle='--')

xlims = ax.get_xlim()
axins.set_xlim(min(xlims), 18)
axins.set_ylim(0, 4)

axins.set_xticks(np.linspace(int(min(ax.get_xlim())), 18, 18 - int(min(ax.get_xlim()))), labels=[])
axins.set_yticks([0, 1, 2, 3, 4], labels=[])

axins.minorticks_on()
axins.yaxis.set_tick_params(which='minor', bottom=False)

ax.set_xlim(0.9 * min(xlims), max(xlims))

ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes("right", size="30%", pad=0.2)

cax.plot(impactor.mass / M0, sim.impactor.z / 1e3, c=cm.bamako(0.6))

if len(sim.fragments):
    cax.plot(impactor.mass[-1] / M0, sim.impactor.z[-1] / 1e3, 'x', c='k', alpha=0.5)

    for fragment in sim.fragments:
        
        cax.plot(fragment.mass / M0, fragment.z / 1e3, c=cm.bamako(0.6))
        if fragment.z[-1] > 1:
            cax.plot(fragment.mass[-1] / M0, fragment.z[-1] / 1e3, 'x', c='k', alpha=0.5)

cax.set_xlim(0, 1)
cax.set_ylim(0, 25)

cax.set_yticklabels([])

ax.set_xlabel(r'Velocity [${\rm km\,s}^{-1}$]', fontsize=13)
ax.set_ylabel(r'Altitude [km]', fontsize=13)
cax.set_xlabel(r'Mass [$M_0$]', fontsize=13)

ax.minorticks_on()
cax.minorticks_on()

plt.tight_layout()

# plt.savefig('zoom_in_panel.pdf', format='pdf')

plt.show()
