import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cmcrameri.cm as cm
from atmosentry.meteoroid import Meteoroid
from atmosentry import Simulation
import corner

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


fig_width, fig_height = set_size('thesis', 1, (1, 1))

rho_com = 0.6e3
rho_atm0 = 1.225

theta0 = 45. * np.pi / 180.
V0 = 20e3

R0 = 1500
M0 = rho_com * (4 * np.pi / 3) * (R0 ** 3)

KE0 = 0.5 * M0 * ((V0 / 1e3) ** 2)

N_simulations = 500

mean_masses2 = []
weighted_vels2 = []
tot_mass2 = []
tot_KE2 = []

for i in range(N_simulations):

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
                         rho=0.6e3,
                         eta=2.5e6)

    sim = Simulation()
    sim.impactor = impactor
    sim.Ch = 0.1
    sim.Nfrag = 2
    sim.integrate()

    masses = []
    KEs = []

    for fragment in sim.fragments:
        if fragment.z[-1] < 1:
            v = np.sqrt(fragment.vx[-1] ** 2 + fragment.vy[-1] ** 2 + fragment.vz[-1] ** 2) / 1e3
            masses.append(fragment.mass[-1])
            KEs.append(0.5 * fragment.mass[-1] * (v ** 2))

            mean_masses2 = np.append(mean_masses2, fragment.mass[-1])
            weighted_vels2 = np.append(weighted_vels2, v)
    
    tot_mass2 = np.append(tot_mass2, np.sum(masses))
    tot_KE2 = np.append(tot_KE2, np.sum(KEs))

mean_masses3= []
weighted_vels3 = []
tot_mass3 = []
tot_KE3 = []

for i in range(N_simulations):

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
                         rho=0.6e3,
                         eta=2.5e6)

    sim = Simulation()
    sim.impactor = impactor
    sim.Ch = 0.1
    sim.Nfrag = 3
    sim.integrate()

    masses = []
    KEs = []

    for fragment in sim.fragments:
        if fragment.z[-1] < 1:
            v = np.sqrt(fragment.vx[-1] ** 2 + fragment.vy[-1] ** 2 + fragment.vz[-1] ** 2) / 1e3
            masses.append(fragment.mass[-1])
            KEs.append(0.5 * fragment.mass[-1] * (v ** 2))

            mean_masses3 = np.append(mean_masses3, fragment.mass[-1])
            weighted_vels3 = np.append(weighted_vels3, v)
    
    tot_mass3 = np.append(tot_mass3, np.sum(masses))
    tot_KE3 = np.append(tot_KE3, np.sum(KEs))

mean_masses4= []
weighted_vels4 = []
tot_mass4 = []
tot_KE4 = []

for i in range(N_simulations):

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
                         rho=0.6e3,
                         eta=2.5e6)

    sim = Simulation()
    sim.impactor = impactor
    sim.Ch = 0.1
    sim.Nfrag = 4
    sim.integrate()

    masses = []
    KEs = []

    for fragment in sim.fragments:
        if fragment.z[-1] < 1:
            v = np.sqrt(fragment.vx[-1] ** 2 + fragment.vy[-1] ** 2 + fragment.vz[-1] ** 2) / 1e3
            masses.append(fragment.mass[-1])
            KEs.append(0.5 * fragment.mass[-1] * (v ** 2))

            mean_masses4 = np.append(mean_masses4, fragment.mass[-1])
            weighted_vels4 = np.append(weighted_vels4, v)
    
    tot_mass4 = np.append(tot_mass4, np.sum(masses))
    tot_KE4 = np.append(tot_KE4, np.sum(KEs))


mean_masses2 = np.array(mean_masses2)
weighted_vels2 = np.array(weighted_vels2)

mean_masses3 = np.array(mean_masses3)
weighted_vels3 = np.array(weighted_vels3)

mean_masses4 = np.array(mean_masses4)
weighted_vels4 = np.array(weighted_vels4)

data2 = np.vstack([mean_masses2 / M0, weighted_vels2]).T
data3 = np.vstack([mean_masses3 / M0, weighted_vels3]).T
data4 = np.vstack([mean_masses4 / M0, weighted_vels4]).T


tot_data2 = np.vstack([tot_mass2 / M0, tot_KE2 / KE0]).T
tot_data3 = np.vstack([tot_mass3 / M0, tot_KE3 / KE0]).T
tot_data4 = np.vstack([tot_mass4 / M0, tot_KE4 / KE0]).T

figure1 = corner.corner(data2, labels=["Mass", "Velocity"],
                       color='tab:blue', show_titles=False, title_fmt=".3f", )

corner.corner(data3, labels=["Mass", "Velocity"],
              show_titles=False, title_fmt=".3f",
              color='tab:orange', plot_datapoints=False, fill_contours=True,
              fig=figure1) 

corner.corner(data4, labels=["Mass", "Velocity"],
              show_titles=False, title_fmt=".3f",
              color='tab:green', plot_datapoints=False, fill_contours=True,
              fig=figure1) 

axes = np.array(figure1.axes).reshape((2, 2))

axes[0, 0].set_xlim(0, 1)
axes[1, 0].set_xlim(0, 1)
axes[1, 0].set_ylim(10, 20)
axes[1, 1].set_xlim(10, 20)

plt.minorticks_on()

figure2 = corner.corner(tot_data2, labels=["Mass", "KE"],
                       color='tab:blue', show_titles=False, title_fmt=".3f", )

corner.corner(tot_data3, labels=["Mass", "KE"],
              show_titles=False, title_fmt=".3f",
              color='tab:orange', plot_datapoints=False, fill_contours=True,
              fig=figure2) 

corner.corner(tot_data4, labels=["Mass", "KE"],
              show_titles=False, title_fmt=".3f",
              color='tab:green', plot_datapoints=False, fill_contours=True,
              fig=figure2) 

axes = np.array(figure2.axes).reshape((2, 2))

axes[0, 0].set_xlim(0, 1)
axes[1, 0].set_xlim(0, 1)
axes[1, 0].set_ylim(0, 1)
axes[1, 1].set_xlim(0, 1)

plt.minorticks_on()

plt.show()
