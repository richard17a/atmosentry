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

R0 = 150
M0 = rho_com * (4 * np.pi / 3) * (R0 ** 3)

N_simulations = 5000

# Store the mean mass and weighted velocity for all simulations
mean_masses2 = []
weighted_vels2 = []

for i in range(N_simulations):
    # First simulation setup
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
    sim.Nfrag = 2
    sim.integrate()

    masses = []
    vels = []

    # Gather fragment information from the simulation
    for fragment in sim.fragments:
        if fragment.z[-1] < 1:
            v = np.sqrt(fragment.vx[-1] ** 2 + fragment.vy[-1] ** 2 + fragment.vz[-1] ** 2) / 1e3
            masses.append(fragment.mass[-1])
            vels.append(v)

    masses = np.array(masses)
    vels = np.array(vels)

    # Calculate mean mass and weighted velocity
    if len(masses) > 0:  # Avoid empty fragments
        mean_mass = np.mean(masses)
        weighted_vel = np.average(vels, weights=masses)

        # Store results for the corner plot
        mean_masses2.append(mean_mass)
        weighted_vels2.append(weighted_vel)


mean_masses3= []
weighted_vels3 = []

for i in range(N_simulations):
    # First simulation setup
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
    sim.Nfrag = 3
    sim.integrate()

    masses = []
    vels = []

    # Gather fragment information from the simulation
    for fragment in sim.fragments:
        if fragment.z[-1] < 1:
            v = np.sqrt(fragment.vx[-1] ** 2 + fragment.vy[-1] ** 2 + fragment.vz[-1] ** 2) / 1e3
            masses.append(fragment.mass[-1])
            vels.append(v)

    masses = np.array(masses)
    vels = np.array(vels)

    # Calculate mean mass and weighted velocity
    if len(masses) > 0:  # Avoid empty fragments
        mean_mass = np.mean(masses)
        weighted_vel = np.average(vels, weights=masses)

        # Store results for the corner plot
        mean_masses3.append(mean_mass)
        weighted_vels3.append(weighted_vel)


mean_masses2 = np.array(mean_masses2)
weighted_vels2 = np.array(weighted_vels2)

mean_masses3 = np.array(mean_masses3)
weighted_vels3 = np.array(weighted_vels3)

data2 = np.vstack([mean_masses2 / M0, weighted_vels2]).T
data3 = np.vstack([mean_masses3 / M0, weighted_vels3]).T

figure = corner.corner(data2, labels=["Mass", "Velocity"],
                       color='tab:blue', show_titles=True, title_fmt=".3f", quantiles=[0.16, 0.5, 0.84])

corner.corner(data3, labels=["Mass", "Velocity"],
              show_titles=True, title_fmt=".3f", quantiles=[0.16, 0.5, 0.84],
              color='tab:orange', plot_datapoints=False, fill_contours=True,
              fig=figure) 

axes = np.array(figure.axes).reshape((2, 2))

axes[0, 0].set_xlim(0, 1)
axes[1, 0].set_xlim(0, 1)
axes[1, 0].set_ylim(10, 20)
axes[1, 1].set_xlim(10, 20)

plt.show()