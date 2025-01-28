# pylint: disable=C0103,C0121,E1101

"""
Script to generate figures 5 and 6 from Anslow+ 2025 (MNRAS, subm.)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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

fig_width, fig_height = set_size('thesis')


def calc_Dmin(V0, rho_atm, rho_com, theta0=45. * np.pi / 180., R0=20):
    """
    Calculate the minimum radius of a meteoroid required to ensure that 
    at least 75% of its initial mass reaches the Earth's surface after passing 
    through the atmosphere.
    """

    M0 = rho_com * (4 * np.pi / 3) * (R0 ** 3)

    flag_ = False
    while flag_ == False:

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

        sim.dt = 1e-0

        sim.rho0 = rho_atm

        sim.impactor = impactor

        sim.integrate()

        frags = [frag.state.mass[-1] for frag in sim.fragments if frag.state.z[-1] < 1]

        if np.sum(frags) < 0.75 * M0:
            R0 = 1.01 * R0
            M0 = rho_com * (4 * np.pi / 3) * (R0 ** 3)
        else:
            flag_ = True

    return R0


def main():
    """
    main plotting method
    """

    rho_comets = np.logspace(-1, 1, 5) * 0.6e3
    rho_comets = rho_comets[::-1]
    r_prev = 20

    r_min_com  = []
    for rho_comet in rho_comets:

        print(r'rho_com=', rho_comet, end='\r')
        r_temp = calc_Dmin(15e3, 1.225, rho_comet, R0=r_prev)
        r_min_com = np.append(r_min_com, r_temp)
        r_prev = r_temp

    rho_atms = np.logspace(-2, 2, 5) * 1.225
    r_prev = 5

    r_min_atm  = []
    for rho_atm in rho_atms:

        print(r'rho_atm=', rho_atm, end='\r')
        r_temp = calc_Dmin(15e3, rho_atm, 0.6e3, R0=r_prev)
        r_min_atm = np.append(r_min_atm, r_temp)
        r_prev = r_temp

    _, ax1 = plt.subplots(figsize=(fig_width, fig_height))

    ax1.plot(rho_comets / 0.6e3, r_min_com, marker='.', c=cm.bamako(0.2))

    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax1.set_xlabel(r'Comet bulk density [$0.6\,{\rm g\,cm}^{-3}$]', fontsize=13, c=cm.bamako(0.2))
    ax1.set_ylabel(r'Minimum diameter [m]', fontsize=13)

    ax2 = ax1.twiny()

    ax2.plot(rho_atms / 1.225, r_min_atm, marker='.', c=cm.bamako(0.7))

    ax2.set_xscale('log')

    ax2.set_xlabel(r'Atmospheric surface density [$1.225\,{\rm kg\,m}^{-3}$]',
                   fontsize=13, c=cm.bamako(0.7))

    ax1.minorticks_on()
    ax2.minorticks_on()

    # plt.savefig('./paper_figures/figures/minimum_cometary_diameter.pdf',
    #             format='pdf', bbox_inches='tight')

    plt.show()


def main_angle():
    """
    main plotting method
    """

    thetas = np.linspace(5, 90, 10)
    thetas = thetas[::-1]

    r_prev = 20

    r_min  = []
    for theta in thetas:

        print(r'theta=', theta, end='\r')

        r_temp = calc_Dmin(15e3, 1.225, 0.6e3, theta * np.pi / 180, r_prev)
        r_min = np.append(r_min, r_temp)
        r_prev = r_temp

    _ = plt.figure(figsize=(fig_width, fig_height))

    plt.plot(thetas, r_min, marker='.', c=cm.bamako(0.3), zorder=10)
    plt.plot(thetas, r_min[0] / np.sin(thetas * np.pi / 180), marker='.',
             c=cm.bamako(0.6), ls='--', label=r'$D_{\rm min} \mathrm{cosec}{\theta}$', zorder=0)

    plt.yscale('log')

    plt.xlim(0, 90)
    plt.ylim(8e1, )

    plt.xlabel(r'Impact angle [deg]', fontsize=13)
    plt.ylabel(r'Minimum diameter [m]', fontsize=13)

    plt.minorticks_on()

    plt.savefig('./paper/figures/minimum_cometary_diameter_angles.pdf',
                format='pdf', bbox_inches='tight')

    plt.show()


if __name__ == "__main__":

    main()
    # main_angle()
