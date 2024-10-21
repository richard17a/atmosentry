"""
Add docstring...
"""

import numpy as np


class Meteoroid():
    """
    This is a meteroid object
    """

    def __init__(self, 
                 x,
                 y,
                 z, 
                 vx, 
                 vy, 
                 vz, 
                 theta, 
                 radius, 
                 mass, 
                 sigma,
                 rho,
                 eta,
                 dM=None,
                 dEkin=None, 
                 children=False):
        
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.theta = theta
        self.radius = radius
        self.mass = mass
        self.sigma = sigma
        self.rho = rho
        self.eta = eta
        self.dM = dM
        self.dEkin = dEkin
        self.children = children

        # should add setters and getters here (and check mass, radius, density are consistent.....)
