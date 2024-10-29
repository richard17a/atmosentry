import numpy as np
import pandas as pd
import multiprocessing as mp
from atmosentry.meteoroid import Meteoroid
from atmosentry import Simulation


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


def run_simulation(R0, M0, theta0, V0, Nfrags):

    np.random.seed()

    impactor = Meteoroid(
        x=0,
        y=0,
        z=100e3,
        vx=-V0 * np.cos(theta0),
        vy=0,
        vz=-V0 * np.sin(theta0),
        radius=R0,
        mass=M0,
        sigma=1e4,
        rho=0.6e3,
        eta=2.5e6
    )

    sim = Simulation()
    sim.impactor = impactor
    sim.Nfrags = Nfrags
    sim.integrate()

    if not len(sim.fragments):

        vel = np.sqrt(sim.impactor.vx[-1] ** 2 + sim.impactor.vy[-1] ** 2 + sim.impactor.vz[-1] ** 2)
        return (0, sim.impactor.mass[-1] / M0, vel / 1e3, 0)
    
    else:

        fragments = [frag for frag in sim.fragments if frag.z[-1] < 1]
        vels, masses = [], []
        for frag in fragments:
            vel = np.sqrt(frag.vx[-1] ** 2 + frag.vy[-1] ** 2 + frag.vz[-1] ** 2)
            vels.append(vel)
            masses.append(frag.mass[-1])

        avg_vel = np.sum(np.array(masses) * np.array(vels)) / np.sum(masses) / 1e3
        return (1, np.mean(masses) / M0, avg_vel, len(fragments))


def main():

    rho_com = 0.6e3
    rho_atm0 = 1.225

    Nfrags = 4

    theta0 = 45. * np.pi / 180.
    V0 = 20e3

    R_small = np.logspace(np.log10(10), np.log10(40), 3)
    R_mid = np.logspace(np.log10(50), np.log10(500), 10)
    R_large = np.logspace(np.log10(600), 4, 7)
    R_sm = np.append(R_small, R_mid)
    radii = np.append(R_sm, R_large)

    fpath = './examples/data/'
    fname = 'output' + str(Nfrags) + '.txt' 

    with open(fpath + fname, 'w') as f:
        f.write("R0 [km]\tfrag_bool\tMf [M0]\tsigma_Mf[M0]\tvel [km/s]\tsigma_vel [km/s]\tNfrags\tsigma_Nfrags\n")

    for i in range(len(radii)):
        M0 = rho_com * (4 * np.pi / 3) * (radii[i] ** 3)

        num_processors = mp.cpu_count()

        with mp.Pool(processes=num_processors) as pool:
            results = pool.starmap(run_simulation, [(radii[i], M0, theta0, V0, Nfrags) for _ in range(num_processors)])

        frag_bool = [res[0] for res in results]
        final_masses = [res[1] for res in results]
        final_velocities = [res[2] for res in results]
        nfrags_list = [res[3] for res in results]

        avg_mass = np.mean(final_masses)
        std_mass = np.std(final_masses)
        avg_velocity = np.mean(final_velocities)
        std_velocity = np.std(final_velocities)
        avg_nfrags = np.mean(nfrags_list)
        std_nfrags = np.std(nfrags_list)

        with open(fpath + fname, 'a') as f:
            f.write(f"{radii[i] / 1e3}\t{max(frag_bool)}\t{avg_mass}\t{std_mass}\t{avg_velocity}\t{std_velocity}\t{avg_nfrags}\t{std_nfrags}\n")


def analysis():

    data = pd.read_fwf('./examples/data/output4.txt', sep='\t', header=None)

    column_names = data.iloc[0, 0].split('\t')
    column_names[0] = '$R_0$ [km]'
    column_names[1] = "Fragmentation"

    data = pd.read_csv('./examples/data/output4.txt', sep='\t', skiprows=1, header=None)

    data.columns = column_names
    # print(data.columns)

    data['$R_0$ [km]'] = data['$R_0$ [km]'].map('{:.2f}'.format)
    data['Mf [M0]'] = data['Mf [M0]'].map('{:.2f}'.format) + ' $\pm$ ' + data['sigma_Mf[M0]'].map('{:.2f}'.format)
    data['vel [km/s]'] = data['vel [km/s]'].map('{:.2f}'.format) + ' $\pm$ ' + data['sigma_vel [km/s]'].map('{:.2f}'.format)
    data['Nfrags'] = data['Nfrags'].map('{:.2f}'.format) + ' $\pm$ ' + data['sigma_Nfrags'].map('{:.2f}'.format)

    data = data.drop(columns=['sigma_Mf[M0]', 'sigma_vel [km/s]', 'sigma_Nfrags'])

    latex_table = data.to_latex(index=False, column_format='|c|'*len(data.columns), escape=False)

    with open('./examples/data/table_combined4.tex', 'w') as f:
        f.write(latex_table)


def get_data(fname):

    data = pd.read_fwf(fname, sep='\t', header=None)

    column_names = data.iloc[0, 0].split('\t')
    column_names[0] = '$R_0$ [km]'
    column_names[1] = "Fragmentation"

    data = pd.read_csv(fname, sep='\t', skiprows=1, header=None)

    data.columns = column_names
    
    R0 = np.array(data['$R_0$ [km]'])
    Mf = np.array(data['Mf [M0]'])
    sigma_Mf = np.array(data['sigma_Mf[M0]'])
    vel = np.array(data['vel [km/s]'])
    sigma_vel = np.array(data['sigma_vel [km/s]'])
    Nfrags = np.array(data['Nfrags'])
    sigma_Nfrags = np.array(data['sigma_Nfrags'])

    return R0, (Mf, sigma_Mf), (vel, sigma_vel), (Nfrags, sigma_Nfrags)


def plot_data():

    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    fig_width, fig_height = set_size('thesis', 1, (1, 1))

    _ = plt.figure(figsize=(fig_width, fig_height))

    R0, (Mf, sigma_Mf), (vel, sigma_vel), (Nfrags, sigma_Nfrags) = get_data('./examples/data/output2.txt')
    plt.errorbar(R0, vel, yerr=sigma_vel, marker='x', capsize=3, label=r'$N_{\rm frag} = 2$')
    # plt.errorbar(R0, Mf, yerr=sigma_Mf, marker='x', capsize=3, label=r'$N_{\rm frag} = 2$')
    # plt.errorbar(R0, Nfrags, yerr=sigma_Nfrags, marker='x', capsize=3, label=r'$N_{\rm frag} = 2$')

    R0, (Mf, sigma_Mf), (vel, sigma_vel), (Nfrags, sigma_Nfrags) = get_data('./examples/data/output3.txt')
    plt.errorbar(R0, vel, yerr=sigma_vel, marker='x', capsize=3, label=r'$N_{\rm frag} = 3$')
    # plt.errorbar(R0, Mf, yerr=sigma_Mf, marker='x', capsize=3, label=r'$N_{\rm frag} = 3$')
    # plt.errorbar(R0, Nfrags, yerr=sigma_Nfrags, marker='x', capsize=3, label=r'$N_{\rm frag} = 3$')

    R0, (Mf, sigma_Mf), (vel, sigma_vel), (Nfrags, sigma_Nfrags) = get_data('./examples/data/output4.txt')
    plt.errorbar(R0, vel, yerr=sigma_vel, marker='x', capsize=3, label=r'$N_{\rm frag} = 4$')
    # plt.errorbar(R0, Mf, yerr=sigma_Mf, marker='x', capsize=3, label=r'$N_{\rm frag} = 4$')
    # plt.errorbar(R0, Nfrags, yerr=sigma_Nfrags, marker='x', capsize=3, label=r'$N_{\rm frag} = 4$')

    plt.xscale('log')

    plt.xticks([0.01, 0.1, 1, 10], labels=[0.01, 0.1, 1, 10])

    plt.minorticks_on()

    plt.xlabel('Initial radius [km]', fontsize=13)
    plt.ylabel(r'$\langle v_{\rm frag} \rangle_m$ [km/s]', fontsize=13)
    # plt.ylabel(r'$\langle m_{\rm frag} \rangle / M_0$', fontsize=13)
    # plt.ylabel('Number of fragments at surface', fontsize=13)

    plt.legend(frameon=False, fontsize=11)

    plt.savefig('./examples/figures/Nfrag_sensitivity_vfrag.pdf', format='pdf', bbox_inches='tight')

    plt.show()


if __name__ == "__main__":

    # main()
    # analysis()
    plot_data()
