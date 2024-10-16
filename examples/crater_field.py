import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cmcrameri.cm as cm
import multiprocessing as mp
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


def calc_D_crater(M_imp, D_imp, v_imp, theta_imp):

    rho_tar = 3e3

    return 1.677 * (M_imp ** 0.113) * (D_imp ** -0.22) * (rho_tar ** (-1/3)) * ((0.5 * M_imp * v_imp ** 2) ** 0.22) * (9.81 ** -0.22) * (np.sin(theta_imp) ** (1/3))


R0 = 100
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

_ = plt.figure(figsize=(fig_height, fig_height))

surface_frags = [frag for frag in sim.fragments if frag.z[-1] < 1]
if len(surface_frags):

    x_coords = [frag.x[-1] for frag in surface_frags]
    y_coords = [frag.y[-1] for frag in surface_frags]

    x_bar = np.mean(x_coords)
    y_bar = np.mean(y_coords)

    for frag in surface_frags:

        v_imp = np.sqrt(frag.vx[-1] ** 2 + frag.vy[-1] ** 2 + frag.vz[-1] ** 2)

        D_crater = calc_D_crater(frag.mass[-1], 2 * frag.radius[-1], v_imp, frag.theta[-1])

        plt.scatter(frag.x[-1] - x_bar, frag.y[-1] - y_bar, marker=None, facecolors='none')
        
        circle = plt.Circle((frag.x[-1] - x_bar, frag.y[-1] - y_bar), D_crater / 2, color='k', fill=False)
        plt.gca().add_patch(circle)

else:

    plt.scatter(impactor.x[-1], impactor.y[-1])

plt.minorticks_on()

x_limits = plt.gca().get_xlim()
y_limits = plt.gca().get_ylim()

max_lims = [max(x_limits), max(y_limits)]

plt.xlim(-max(max_lims), max(max_lims))
plt.ylim(-max(max_lims), max(max_lims))

plt.xlabel(r'$x$ [m]', fontsize=13)
plt.ylabel(r'$y$ [m]', fontsize=13)

plt.savefig('crater_field.pdf', bbox_inches='tight', format='pdf')

plt.show()
