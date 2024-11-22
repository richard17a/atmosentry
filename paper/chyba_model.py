import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cmcrameri.cm as cm
from atmosentry.meteoroid import Meteoroid
from atmosentry import Simulation
from scipy.integrate import solve_ivp

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


def differential_equations(t: float, y: list, sigma_imp: float, rho_imp: float, eta: float, rho_atm0: float):
    """
    Defining the differential equations to be solved when calculating the comets trajectory
    """
 
    ### ------ Defining constants ------
    C_d = 0.7 # the comet's drag coefficient
    C_h = 0.02 # comet's heat transfer coefficient (fraction of energy that heats comet vs atmosphere)
    C_l = 0.001 # the comet's lift coefficient
    M_E = 5.97e24
    R_E = 6371e3
    G = 6.67e-11
    sigma = 5.6704e-8
    T = 25000 # assumed temperatuer of shocked gas at leading edge of the comet
    H = 7.2e3 # Earth's atmospheric scale height
    ### ----------------------- ###
 
    V, M, theta, Z, R, Rdot = y
 
    rho_a = rho_atm0 * np.exp(- Z / H)
    A = np.pi * R ** 2
    g = G * M_E / (R_E + Z) ** 2
 
    # equation 1 chyba et al. 1993 (I actually think there is a typo in their manuscript, and should read + g sin(theta))
    dVdt = - 0.5 * C_d * rho_a * A * V**2 / M + g * np.sin(theta)
 
    # equation 4 chyba et al. 1993
    dthetadt = (M * g * np.cos(theta) - 0.5 * C_l * rho_a * A * V**2) /\
               (M * V) - V * np.cos(theta) / (R_E + Z)
 
    # evaluate the evolution of comet's altitude (vertical component of velocity vector)
    dZdt = - V * np.sin(theta)
 
    # equation 3 chyba et al. 1993
    dMdt = - np.minimum(sigma * T**4, 0.5 * C_h * rho_a * V**3) * A / eta
 
    if 0.5 * C_d * rho_a * V**2 > sigma_imp:
        R_dot = Rdot
        R_ddot = C_d * rho_a * V**2 / (2 * rho_imp * R)
    else:
        R_dot = 0.0
        R_ddot = 0.0
 
    return [dVdt, dMdt, dthetadt, dZdt, R_dot, R_ddot]
 
def event_Z_crossing(t: float, y: list):
    """
    Event triggered when altitude crosses 0 (i.e. the comet hits the ground)
    """
 
    return y[3]
 
def event_mass_zero(t: float, y: list):
    """
    Event triggered when all mass has been ablated
    """
 
    return y[1]
 
def event_pancake(t: float, y: list, R0: float):
    """
    Event triggered when comet's radius (size of pancake) exceeds 6 * initial radius
    At this point the model seems to break down - in reality individual fragments would
    form and develop their own bow shocks - so I just halt the integration at this point
    and classify it as an airburst event.
    """
 
    return 6 * R0 - y[4]
 
def event_dVdt_zero(t: float, y: list, rho_atm0: float):
    """
    Event triggered when object reaches terminal velocity
    """
 
    C_d = 0.7 # the comet's drag coefficient
    H = 7.2e3 # Earth's atmospheric scale height
    G = 6.67e-11
    M_E = 5.97e24
    R_E = 6371e3
 
    V, M, theta, Z, R, _ = y
    rho_a = rho_atm0 * np.exp(- Z / H) # assuming an isothermal atmosphere
    A = np.pi * R ** 2 # surface area of leading edge of comet
    g = G * M_E / (R_E + Z) ** 2
 
    term_vel = np.sqrt((M * g * np.sin(theta)) / (C_d * rho_a * A))
 
    out_num = 1
 
    if (np.isclose(term_vel, V, rtol=1e-01)) & (V < 1e3):
        """
        included the requirement for v<1km/s, as terminal velocity is initially
        very low due to low density in upper atmos
        """
        print(term_vel, V)
        out_num = 0
 
    return out_num
 
def run_intergration_chyba(V0: float, M0: float, theta0: float, Z0: float, R0: float, Rdot0: float, sigma_imp: float, rho_imp: float, eta: float, rho_atm0=1.225):
    """
    This function will run the integration and calculate the atmospheric trjectory of the comet

    Args:
        V0: Initial velocity of comet
        M0: Initial mass of comet
        theta0: Initial angle of comet's trajectory
        Z0: Initial altitude of comet
        R0: Intiial radius of comet
        Rdot0: Initial rate of deformation (this is always zero...!)
        sigma_imp: Tensile strength of comet
        rho_imp: Bulk density of comet
        eta: Comet's heat of ablation
        rho_atmo0: Density of atmosphere at altitude=0
    """
 
    # Time span for the integration (definitely doesn't need to be this large!)
    t_span = (0, 500)
 
    def event_pancake_with_R0(t: float, y: list):
        """
        Event triggered when size of pancake exceeds 6 * initial radius
        """
        return event_pancake(t, y, R0)

    def event_dVdt_zero_rhoatm0(t: float, y: list):
        """
        Event triggered when object reaches terminal velocity
        """
        return event_dVdt_zero(t, y, rho_atm0)
 
    event_Z_crossing.terminal = True
    event_Z_crossing.direction = -1
 
    event_mass_zero.terminal = True
    event_mass_zero.direction = -1
 
    event_dVdt_zero_rhoatm0.terminal = True
    event_dVdt_zero_rhoatm0.direction = 0

    event_pancake_with_R0.terminal = True
    event_pancake_with_R0.direction = -1

    # these events terminate the integration (i.e. when the comet's mass = 0, or the altitude = 0 etc.)
    events = [event_Z_crossing, event_mass_zero, event_dVdt_zero_rhoatm0, event_pancake_with_R0]

    # Here we use solve_ivp from scipy to solve differential equations
    sol_iso = solve_ivp(
        fun=lambda t, y: differential_equations(t, y, sigma_imp, rho_imp, eta, rho_atm0),
        t_span=t_span,
        y0=[V0, M0, theta0, Z0, R0, Rdot0],
        method='RK45',
        dense_output=True,
        events=events,
        max_step=1e-2  # Maximum step size of the integration
    )

    # extract the solutions for the comet's trajectory - i.e. all times until the integration is terminated
    t = sol_iso.t

    vel = sol_iso.sol(t)[0][:len(t)]
    mass = sol_iso.sol(t)[1][:len(t)]
    theta = sol_iso.sol(t)[2][:len(t)]
    altitude = sol_iso.sol(t)[3][:len(t)]
    radius = sol_iso.sol(t)[4][:len(t)]

    C_d = 0.7 # comet's draf coefficient
    C_h = 0.02 # comet's heat transfer coefficient (fraction of energy that heats comet vs atmosphere)
    M_E = 5.97e24
    R_E = 6371e3
    G = 6.67e-11
    sigma = 5.6704e-8
    T = 25000 # assumed temperatuer of shocked gas at leading edge of the comet
    H = 7.2e3 # Earth's atmospheric scale height

    g = G * M_E / (R_E + altitude) ** 2
    rho_a = rho_atm0 * np.exp(- altitude / H)
    A = np.pi * radius ** 2

    # equation 1 chyba et al. 1993 (I actually think there is a typo in their manuscript, and should read + g sin(theta))
    dVdt = - 0.5 * C_d * rho_a * A * vel**2 / mass

    # equation 3 chyba et al. 1993
    dMdt = - np.minimum(sigma * T**4, 0.5 * C_h * rho_a * vel**3) * A / eta

    # evaluate the altitude evolution of the comet's trajectory
    dZdt = - vel * np.sin(theta)

    ram_pressure = 0.7 * rho_atm0 * np.exp(-altitude / H) * vel ** 2 / 2

    # calculate the rate of change of the comet's kinetic energy (wrt time, and altitude respectively)
    Ekindot = mass * vel * dVdt + 0.5 * vel**2 * dMdt
    dEkindh = Ekindot / dZdt

    return t, vel, mass, theta, altitude, radius, ram_pressure, dEkindh


if __name__ == "__main__":
    rho_com = 0.6e3
    rho_atm0 = 1.225

    theta0 = 45. * np.pi / 180.
    V0 = 20e3

    R0 = [10, 50, 150, 500, 1000]
    R0 = [50, 150, 500, 1000]
    M0 = rho_com * (4 * np.pi / 3) * (np.array(R0) ** 3)

    _ = plt.figure(figsize=(fig_width, fig_height))

    for i in range(len(R0)):

        impactor = Meteoroid(x=0,
                            y=0,
                            z=100e3,
                            vx=-V0 * np.cos(theta0),
                            vy=0,
                            vz=-V0 * np.sin(theta0),
                            theta=theta0,
                            radius=R0[i],
                            mass=M0[i],
                            sigma=1e4,
                            rho=0.6e3,
                            eta=2.5e6)

        sim = Simulation()
        sim.impactor = impactor
        sim.integrate()

        vel = np.sqrt(sim.impactor.vx ** 2 + sim.impactor.vy ** 2 + sim.impactor.vz ** 2)

        plt.plot(vel / 1e3, sim.impactor.z / 1e3, c=cm.bamako((len(R0) - i) / len(R0)), label=fr'$R_0=$ {R0[i]} m')
        if len(sim.fragments):
            plt.plot(vel[-1] / 1e3, sim.impactor.z[-1] / 1e3, 'x', c='k')

            for fragment in sim.fragments:

                vel = np.sqrt(fragment.vx ** 2 + fragment.vy ** 2 + fragment.vz ** 2)

                plt.plot(vel / 1e3, fragment.z / 1e3, c=cm.bamako((len(R0) - i) / len(R0)), )
                plt.plot(vel[-1] / 1e3, fragment.z[-1] / 1e3, 'x', c='k', alpha=0.5)

        _, vel, mass, _, altitude, _, _, dEkin_dh =\
                run_intergration_chyba(V0, M0[i], theta0, 100e3, R0[i], 0, 1e4, rho_com, 2.5e6)

        plt.plot(vel / 1e3, altitude / 1e3, color=cm.bamako((len(R0) - i - 0.5) / len(R0)), ls='--')

    plt.ylim(0, 60)

    plt.xlabel(r'Velocity [km/s]', fontsize=13)
    plt.ylabel(r'Altitude [km]', fontsize=13)

    plt.legend(frameon=False, fontsize=11)

    plt.minorticks_on()

    plt.show()
