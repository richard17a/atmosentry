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


def simulate_mass_loss(R0, rho_com, V0, theta0):

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

    if len(sim.fragments):
        frag_mass = sum([frag.mass[-1] for frag in sim.fragments if frag.z[-1] < 1])
        return 1 - frag_mass / M0
    
    else:
        return 1 - impactor.mass[-1] / M0


def simulate_mass_loss_for_velocities(R0, rho_com, V0, theta0):

    results = []
    for vel in V0:
        results.append(simulate_mass_loss(R0, rho_com, vel, theta0))

    return results


def parallel_simulation(R0, rho_com, V0, theta0):

    pool = mp.Pool(mp.cpu_count())
    
    tasks = [(rad / 2, rho_com, V0, theta0) for rad in R0]
    
    results = pool.starmap(simulate_mass_loss_for_velocities, tasks)
    
    pool.close()
    pool.join()

    return np.array(results)


if __name__ == "__main__":

    R0 = np.logspace(1, 4, 25)
    vels = [15e3, 20e3, 25e3]

    rho_com = 0.6e3
    theta0 = 45 * np.pi / 180

    results = parallel_simulation(R0, rho_com, vels, theta0)

    final_mass15 = results[:, 0]
    final_mass20 = results[:, 1]
    final_mass25 = results[:, 2]

    _ = plt.figure(figsize=(fig_width, fig_height))

    plt.plot(R0 / 1e3, 100 * final_mass15, c=cm.bamako(0.2), marker='.', label=r'$v_0 = 15\,{\rm km\,s}^{-1}$')
    plt.plot(R0 / 1e3, 100 * final_mass20, c=cm.bamako(0.5), marker='.', label=r'$v_0 = 20\,{\rm km\,s}^{-1}$')
    plt.plot(R0 / 1e3, 100 * final_mass25, c=cm.bamako(0.8), marker='.', label=r'$v_0 = 25\,{\rm km\,s}^{-1}$')

    plt.xlabel('Diameter [km]', fontsize=13)
    plt.ylabel('Fraction mass ablated [%]', fontsize=13)

    plt.xlim(1e-2, 1e1)
    plt.ylim(0, 100)

    plt.xscale('log')

    plt.xticks([1e-2, 1e-1, 1e0, 1e1], labels=[0.01, 0.1, 1, 10])

    plt.minorticks_on()

    plt.legend(frameon=False)

    plt.savefig('fraction_ablated.pdf', bbox_inches='tight', format='pdf')

    plt.show()
