import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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


def calc_Dmin(V0, rho_atm, rho_com):

    theta0 = 45. * np.pi / 180.

    R0 = 10
    M0 = rho_com * (4 * np.pi / 3) * (R0 ** 3) # initial mass of comet

    flag_ = False
    while flag_ == False:

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

        sim.rho0 = rho_atm

        sim.impactor = impactor
        sim.fragments_track = False

        sim.integrate()

        altitude = sim.impactor.z

        if altitude[-1] > 1:
            R0 = 1.01 * R0
            M0 = rho_com * (4 * np.pi / 3) * (R0 ** 3)
        else:
            flag_ = True

    return R0


def main():
    
    rho_comets = np.logspace(-1, 1, 5) * 0.6e3

    r_min_com  = []
    for rho_comet in rho_comets:

        # print(rho_comet)
        r_min_com = np.append(r_min_com, calc_Dmin(20e3, 1.225, rho_comet))

    rho_atms = np.logspace(-2, 2, 5) * 1.225

    r_min_atm  = []
    for rho_atm in rho_atms:

        # print(rho_atm)
        r_min_atm = np.append(r_min_atm, calc_Dmin(20e3, rho_atm, 0.6e3))

    _, ax1 = plt.subplots(figsize=(fig_width, fig_height))

    ax1.plot(rho_comets / 0.6e3, r_min_com, marker='.', c=cm.bamako(0.3))

    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax1.set_xlabel(r'Comet bulk density [$0.6\,{\rm g\,cm}^{-3}$]', fontsize=13, c=cm.bamako(0.3))
    ax1.set_ylabel(r'Minimum diameter [m]', fontsize=13)
    
    ax2 = ax1.twiny()
    
    ax2.plot(rho_atms / 1.225, r_min_atm, marker='.', c=cm.bamako(0.6))

    ax2.set_xscale('log')

    ax2.set_xlabel(r'Atmospheric surface density [$1.225\,{\rm kg\,m}^{-3}$]', fontsize=13, c=cm.bamako(0.6))

    ax1.minorticks_on()
    ax2.minorticks_on()

    plt.savefig('minimum_cometary_diameter.pdf', format='pdf', bbox_inches='tight')

    plt.show()


if __name__ == "__main__":

    main()
