"""
Add docstring...
"""

import numpy as np
from scipy.integrate import solve_ivp
from atmosentry.meteoroid import Meteoroid


def differential_equations(t: float,
                           y: list,
                           sigma_imp: float,
                           rho_imp: float,
                           eta: float,
                           C_d: float,
                           C_h: float,
                           C_l: float,
                           R_pl: float,
                           M_pl: float,
                           rho_atm0: float,
                           H: float):
    
    """
    Defining the differential equations to be solved when calculating the comets trajectory
    """

    ### ------ Defining constants ------ 
    G = 6.67e-11
    sigma = 5.6704e-8
    T = 25000 # assumed temperature of shocked gas at leading edge of the comet
    ### ----------------------- ###

    vx, vy, vz, M, theta, x, y, z, R, Rdot, N = y
    
    v = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

    rho_a = rho_atm0 * np.exp(- z / H)
    A = np.pi * R ** 2
    g = G * M_pl / (R_pl + z) ** 2

    dvxdt = - 0.5 *  C_d * rho_a * A * v * vx / M
    dvydt = - 0.5 *  C_d * rho_a * A * v * vy / M
    dvzdt = - 0.5 *  C_d * rho_a * A * v * vz / M - g

    dthetadt = (M * g * np.cos(theta) - 0.5 * C_l * rho_a * A * v ** 2) /\
               (M * v) - v * np.cos(theta) / (R_pl + z)

    dxdt = vx
    dydt = vy
    dzdt = vz

    dMdt = - np.minimum(sigma * T**4, 0.5 * C_h * rho_a * v**3) * A / eta

    if 0.5 * C_d * rho_a * v ** 2 > sigma_imp:
        R_dot = Rdot
        R_ddot = C_d * rho_a * v ** 2 / (2 * rho_imp * R)

        dNdt = (v / R) * ( (3 * C_d * np.pi * rho_a) / (8 * rho_imp)) ** 0.5
    else:
        R_dot = 0.0
        R_ddot = 0.0
        
        dNdt = 0

    return [dvxdt, dvydt, dvzdt, dMdt, dthetadt, dxdt, dydt, dzdt, R_dot, R_ddot, dNdt]


def event_Z_crossing(t: float, y: list):
    """
    Event triggered when altitude crosses 0 (i.e. the comet hits the ground)
    """

    return y[7]


def event_mass_zero(t: float, y: list):
    """
    Event triggered when all mass has been ablated
    """

    return y[3]


def event_N_crit(t: float, y: list, N_c: float):
    """
    Docstring
    """

    return N_c - y[10]


def event_dVdt_zero(t: float,
                    y: list,
                    rho_atm0: float,
                    C_d: float,
                    M_pl: float,
                    R_pl: float,
                    H: float):
    """
    Event triggered when object reaches terminal velocity
    """

    G = 6.67e-11

    vx, vy, vz, M, theta, x, _, z, R, _, _ = y
    v = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

    rho_a = rho_atm0 * np.exp(- z / H) 
    A = np.pi * R ** 2 
    g = G * M_pl / (R_pl + z) ** 2

    term_vel = np.sqrt((M * g * np.sin(theta)) / (C_d * rho_a * A))

    out_num = 1

    # if (np.isclose(term_vel, v, rtol=1e-01)):
    if (np.isclose(term_vel, v, rtol=1e-01)) & (v < 1e3):
        """
        included the terminal velocity check
        """
        out_num = 0
        # NEED TO ADD A TERMINAL VELOCITY CHECk IN MAIN SIMULATION ROUTINE -- CAN USE TERMINAL VELOCITY EQUATION DOWN TO THE SURFACE...

    return out_num


def run(impactor: Meteoroid,
        C_d: float,
        C_h: float,
        C_l: float,
        R_pl: float,
        M_pl: float,
        rho_atm0: float,
        H: float,
        N_c=2.):
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

    t_span = (0, 5000)

    def event_N_crit_with_Nc(t: float, y: list):
        """
        Event triggered when size of pancake exceeds 6 * initial radius
        """

        return event_N_crit(t, y, N_c)
    
    def event_dVdt_zero_rhoatm0(t: float, y: list):
        """
        Event triggered when object reaches terminal velocity
        """

        return event_dVdt_zero(t, y, rho_atm0, C_d, M_pl, R_pl, H)

    event_Z_crossing.terminal = True
    event_Z_crossing.direction = -1

    event_mass_zero.terminal = True
    event_mass_zero.direction = -1

    event_dVdt_zero_rhoatm0.terminal = True
    event_dVdt_zero_rhoatm0.direction = 0

    event_N_crit_with_Nc.terminal = True
    event_N_crit_with_Nc.direction = -1

    # these events terminate the integration (i.e. when the comet's mass = 0, or the altitude = 0 etc.)
    # events = [event_Z_crossing, event_mass_zero, event_dVdt_zero_rhoatm0, event_N_crit_with_Nc]
    events = [event_Z_crossing, event_mass_zero, event_N_crit_with_Nc]

    x0, y0, z0 = impactor.x, impactor.y, impactor.z
    vx0, vy0, vz0 = impactor.vx, impactor.vy, impactor.vz
    theta0 = impactor.theta
    R0 = impactor.radius
    M0 = impactor.mass
    Rdot0 = 0
    
    rho_imp = impactor.rho
    sigma_imp = impactor.sigma
    eta = impactor.eta

    sol_iso = solve_ivp(
        fun=lambda t, y: differential_equations(t, y, sigma_imp, rho_imp, eta, C_d,
                                                C_h, C_l, R_pl, M_pl, rho_atm0, H),
        t_span=t_span,
        y0=[vx0, vy0, vz0, M0, theta0, x0, y0, z0, R0, Rdot0, 0.],
        method='RK45',
        dense_output=True,
        events=events,
        max_step=1e-2
    )

    t = sol_iso.t

    vx = sol_iso.sol(t)[0][:len(t)]
    vy = sol_iso.sol(t)[1][:len(t)]
    vz = sol_iso.sol(t)[2][:len(t)]
    mass = sol_iso.sol(t)[3][:len(t)]
    theta = sol_iso.sol(t)[4][:len(t)]
    x = sol_iso.sol(t)[5][:len(t)]
    y = sol_iso.sol(t)[6][:len(t)]
    z = sol_iso.sol(t)[7][:len(t)]
    radius = sol_iso.sol(t)[8][:len(t)]
    N_RT = sol_iso.sol(t)[10][:len(t)]

    vel = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    Ekin = 0.5 * mass * (vel ** 2)

    dM = np.abs(np.diff(mass, append=mass[-1]))
    dEkin = np.abs(np.diff(Ekin, append=Ekin[-1]))

    return t, mass, theta, radius, dM, dEkin, x, y, z, vx, vy, vz, N_RT
