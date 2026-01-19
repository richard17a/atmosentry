# pylint: disable=C0103

"""
Script to generate figure 2 from Anslow+ 2025 (MNRAS, subm.)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
                        radius=R0,
                        mass=M0,
                        sigma=1e4,
                        rho=rho_com,
                        eta=2.5e6)

sim = Simulation()
sim.impactor = impactor
sim.integrate()

_, ax = plt.subplots(figsize=(1.25 * fig_width, fig_height))

vel = np.sqrt(sim.impactor.state.vx ** 2 + sim.impactor.state.vy ** 2 + sim.impactor.state.vz ** 2)

ax.plot(vel / 1e3, sim.impactor.state.z / 1e3, color='black')

colors = np.zeros(len(sim.fragments))

if len(sim.fragments):
    ax.plot(vel[-1] / 1e3, sim.impactor.state.z[-1] / 1e3, 'x', c='k', alpha=0.5)

    counter = 0

    for fragment in sim.fragments:

        vel = np.sqrt(fragment.state.vx ** 2 + fragment.state.vy ** 2 + fragment.state.vz ** 2)

        if counter == 0:
            ax.plot(vel / 1e3, fragment.state.z / 1e3, color='tab:green', alpha=1)
        elif counter == 1:
            ax.plot(vel / 1e3, fragment.state.z / 1e3, color='tab:red', alpha=1)
        elif counter == 2:
            ax.plot(vel / 1e3, fragment.state.z / 1e3, color='#2CA083')
        elif counter == 3:
            ax.plot(vel / 1e3, fragment.state.z / 1e3, color='#83A02C')
        elif counter == 4:
            ax.plot(vel / 1e3, fragment.state.z / 1e3, color='#D67D26')
        elif counter == 5:
            ax.plot(vel / 1e3, fragment.state.z / 1e3, color='#D6267F')

        counter += 1

        if fragment.state.z[-1] > 1:
            if fragment.children:
                ax.plot(vel[-1] / 1e3, fragment.state.z[-1] / 1e3, 'x', c='k', alpha=0.5)

ax.set_yscale('log')
ax.set_ylim(1e-1, 100)

ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes("right", size="30%", pad=0.2)

cax.plot(impactor.state.mass / M0, sim.impactor.state.z / 1e3, color='black')
if len(sim.fragments):
    cax.plot(impactor.state.mass[-1] / M0, sim.impactor.state.z[-1] / 1e3, 'x', c='k', alpha=0.5)

    counter = 0

    for fragment in sim.fragments:

        if counter == 0:
            cax.plot(fragment.state.mass / M0, fragment.state.z / 1e3, color='tab:green', alpha=1)
        elif counter == 1:
            cax.plot(fragment.state.mass / M0, fragment.state.z / 1e3, color='tab:red', alpha=1)
        elif counter == 2:
            cax.plot(fragment.state.mass / M0, fragment.state.z / 1e3, color='#2CA083')
        elif counter == 3:
            cax.plot(fragment.state.mass / M0, fragment.state.z / 1e3, color='#83A02C')
        elif counter == 4:
            cax.plot(fragment.state.mass / M0, fragment.state.z / 1e3, color='#D67D26')
        elif counter == 5:
            cax.plot(fragment.state.mass / M0, fragment.state.z / 1e3, color='#D6267F')

        counter += 1

        if fragment.state.z[-1] > 1:
            if fragment.children:
                cax.plot(fragment.state.mass[-1] / M0, fragment.state.z[-1] / 1e3, 'x',
                         c='k', alpha=0.5)

cax.set_xlim(0, 1)

cax.set_yscale('log')
cax.set_ylim(1e-1, 100)

cax.set_yticklabels([])

ax.set_xlabel(r'Velocity [${\rm km\,s}^{-1}$]', fontsize=13)
ax.set_ylabel(r'Altitude [km]', fontsize=13)
cax.set_xlabel(r'Mass [$M_0$]', fontsize=13)

ax.minorticks_on()
cax.minorticks_on()

plt.tight_layout()

plt.savefig('./paper_figures/figures/zoom_in_panel.pdf', format='pdf')

plt.show()
