"""
Add docstring...
"""

import numpy as np
from atmosentry.meteoroid import Meteoroid


def gen_fragment_masses(m_init: float, N_frags: int):
    """"
    Docstring
    """
    
    masses = np.random.rand(N_frags)
    masses = masses / np.sum(masses) * m_init

    return masses


def calc_frag_tensile_strength(m_init: float, m_frags: list, sigma_init: float, alpha: float, beta: float):
    """
    Docstring
    """
    
    x = np.random.normal(0, beta, len(m_frags))

    sigma_frags = sigma_init * ((m_init / m_frags) ** alpha) * (10 ** x)
    
    return sigma_frags


def calc_fragment_velocities(rho_atm: float, rho_imp: float, vel: float, m_init: float, frag_masses: list):
    """
    Docstring
    """
    
    vs = vel * np.sqrt(rho_atm / rho_imp)
    
    phi = 2 * np.pi * np.random.rand(len(frag_masses))
    
    v_alpha = vs * (np.cos(phi) - np.sum(frag_masses * np.cos(phi)) / m_init)
    v_beta = vs * (np.sin(phi) - np.sum(frag_masses * np.sin(phi)) / m_init)
    
    return v_alpha, v_beta


def generate_fragments(fragment: Meteoroid, 
                       rho_atm: float,
                       H: float,
                       alpha: float,
                       beta: float,
                       N_frag: int):
    
    vx, vy, vz = fragment.vx, fragment.vy, fragment.vz
    x, y, z = fragment.x, fragment.y, fragment.z
    mass, theta = fragment.mass, fragment.theta

    rho_m = fragment.rho

    velocity = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

    masses = gen_fragment_masses(mass[-1], N_frag)
    sigma_frags = calc_frag_tensile_strength(mass[-1], masses, fragment.sigma, alpha, beta) # have suspicion this is making them smaller more often than not...

    v_alpha, v_beta = calc_fragment_velocities(rho_atm * np.exp(-z[-1]/H), rho_m, velocity[-1], mass[-1], masses)

    h_final = mass[-1] / (rho_m * np.pi * fragment.radius[-1] ** 2)

    fragments = []
    for counter, _ in enumerate(masses):
        
        # r_frag = (3 * masses[counter] / (4 * rho_m * np.pi)) ** (1./3.)
        r_frag = np.sqrt(masses[counter] / (rho_m * np.pi * h_final))
        # Ok korycansky and Zahnle model fragments as cylinders -- NOT spheres. need to update this (and work out why it stops code running)

        v_frag_x = vx[-1] + v_beta[counter] * np.cos(theta[-1])
        v_frag_y = vy[-1] + v_alpha[counter]
        v_frag_z = vz[-1] + v_beta[counter] * np.sin(theta[-1])

        frag = Meteoroid(x=x[-1], y=y[-1], z=z[-1],
                         vx=v_frag_x, vy=v_frag_y, vz=v_frag_z,
                         theta=theta[-1], radius=r_frag, mass=masses[counter],
                         rho=rho_m, sigma=sigma_frags[counter], eta=fragment.eta)

        fragments = np.append(frag, fragments)

    return fragments
