import numpy as np
import pandas as pd
import multiprocessing as mp
from atmosentry.meteoroid import Meteoroid
from atmosentry import Simulation


def run_simulation(R0, M0, theta0, V0, Nfrags):

    np.random.seed()

    impactor = Meteoroid(
        x=0,
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

    radii = np.logspace(1, 4, 10)

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
    print(data.columns)

    data['$R_0$ [km]'] = data['$R_0$ [km]'].map('{:.2f}'.format)
    data['Mf [M0]'] = data['Mf [M0]'].map('{:.2f}'.format) + ' $\pm$ ' + data['sigma_Mf[M0]'].map('{:.2f}'.format)
    data['vel [km/s]'] = data['vel [km/s]'].map('{:.2f}'.format) + ' $\pm$ ' + data['sigma_vel [km/s]'].map('{:.2f}'.format)
    data['Nfrags'] = data['Nfrags'].map('{:.2f}'.format) + ' $\pm$ ' + data['sigma_Nfrags'].map('{:.2f}'.format)

    data = data.drop(columns=['sigma_Mf[M0]', 'sigma_vel [km/s]', 'sigma_Nfrags'])

    latex_table = data.to_latex(index=False, column_format='|c|'*len(data.columns), escape=False)

    with open('./examples/data/table_combined4.tex', 'w') as f:
        f.write(latex_table)


if __name__ == "__main__":

    # main()
    analysis()
