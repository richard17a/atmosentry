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


R0 = 2500
V0 = 20e3

rho_com = 0.6e3
theta0 = 45 * np.pi / 180

M0 = rho_com * (4 * np.pi / 3) * (R0 ** 3)

Chs = [0.001, 0.01, 0.1, 0.5]

_ = plt.figure(figsize=(fig_width, fig_height))

for i, Ch in enumerate(Chs):

    sim = Simulation()

    impactor = Meteoroid(x=0,
                        y=0,
                        z=150e3,
                        vx=-V0 * np.cos(theta0),
                        vy=0,
                        vz=-V0 * np.sin(theta0),
                        theta=theta0,
                        radius=R0,
                        mass=M0,
                        sigma=1e4,
                        rho=rho_com,
                        eta=2.5e6)

    sim.impactor = impactor
    sim.Ch = Ch
    sim.integrate()

    altitudes = np.linspace(0, 150e3, 1000)
    cumulative_mass_deposition = np.zeros_like(altitudes)

    for j, h in enumerate(altitudes):
        cumulative_dM = 0.0

        cumulative_dM += np.sum(sim.impactor.dM[sim.impactor.z >= h])

        for fragment in sim.fragments:
            cumulative_dM += np.sum(fragment.dM[fragment.z >= h])
        
        cumulative_mass_deposition[j] = cumulative_dM
    
    P_atm = 1. * np.exp(-altitudes / 7.2e3)

    plt.plot(cumulative_mass_deposition * 1e3, P_atm, c=cm.bamako((len(Chs) - i) / len(Chs)), label=rf'$C_H=$ {Ch}')

plt.gca().invert_yaxis()

plt.xscale('log')
plt.yscale('log')

plt.xlim(1e9, 1e15)
plt.ylim(1, 1e-7)

plt.xlabel('Cumulative mass loss [kg]', fontsize=13)
plt.ylabel('Pressure [bar]', fontsize=13)

plt.legend(frameon=False)

# plt.savefig('mass_ablation_CH_comparison.pdf', bbox_inches='tight', format='pdf')

plt.show()
