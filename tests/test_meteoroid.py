"""
Module docstring: COME BACK AND UPDATE WHEN ALL THE SETTERS/GETTERS IN PLACE FOR METEOROID CLASS!
"""

import pytest
import numpy as np
from atmosentry.meteoroid import Meteoroid


def test_meteoroid():
    """
    Docstring
    """

    meteoroid = Meteoroid(x=0,
                          y=0,
                          z=100e3,
                          vx=-20e3 * np.cos(45 * np.pi / 180),
                          vy=0,
                          vz=-20e3 * np.sin(45 * np.pi / 180),
                          radius=150,
                          mass=0.6e3 * (4 * np.pi / 3) * (150 ** 3),
                          sigma=1e4,
                          rho=0.6e3,
                          eta=2.5e6)

    assert meteoroid
    assert (meteoroid.x == 0) and (meteoroid.y == 0) and (meteoroid.z == 100e3)
    assert (meteoroid.vx == -20e3 * np.cos(45 * np.pi / 180)) and\
            (meteoroid.vy == 0) and\
            (meteoroid.vz == -20e3 * np.sin(45 * np.pi / 180))
    assert meteoroid.radius == 150
    assert meteoroid.mass == 0.6e3 * (4 * np.pi / 3) * (150 ** 3)
    assert meteoroid.sigma == 1e4
    assert meteoroid.rho == 0.6e3
    assert meteoroid.eta == 2.5e6


def test_meteoroid_setter_pos():
    """
    Docstring
    """

    with pytest.raises(TypeError):
        _ = Meteoroid(x=[0, 1, 2, 3],
                        y=0,
                        z=100e3,
                        vx=-20e3 * np.cos(45 * np.pi / 180),
                        vy=0,
                        vz=-20e3 * np.sin(45 * np.pi / 180),
                        radius=150,
                        mass=0.6e3 * (4 * np.pi / 3) * (150 ** 3),
                        sigma=1e4,
                        rho=0.6e3,
                        eta=2.5e6)

    with pytest.raises(TypeError):
        _ = Meteoroid(x=0,
                        y=[0, 1, 2, 3],
                        z=100e3,
                        vx=-20e3 * np.cos(45 * np.pi / 180),
                        vy=0,
                        vz=-20e3 * np.sin(45 * np.pi / 180),
                        radius=150,
                        mass=0.6e3 * (4 * np.pi / 3) * (150 ** 3),
                        sigma=1e4,
                        rho=0.6e3,
                        eta=2.5e6)

    with pytest.raises(TypeError):
        _ = Meteoroid(x=0,
                        y=0,
                        z=[0, 1, 2, 3],
                        vx=-20e3 * np.cos(45 * np.pi / 180),
                        vy=0,
                        vz=-20e3 * np.sin(45 * np.pi / 180),
                        radius=150,
                        mass=0.6e3 * (4 * np.pi / 3) * (150 ** 3),
                        sigma=1e4,
                        rho=0.6e3,
                        eta=2.5e6)


def test_meteoroid_setter_vel():
    """
    Docstring
    """

    with pytest.raises(TypeError):
        _ = Meteoroid(x=0,
                        y=0,
                        z=100e3,
                        vx=[0, 1, 2, 3],
                        vy=0,
                        vz=-20e3 * np.sin(45 * np.pi / 180),
                        radius=150,
                        mass=0.6e3 * (4 * np.pi / 3) * (150 ** 3),
                        sigma=1e4,
                        rho=0.6e3,
                        eta=2.5e6)

    with pytest.raises(TypeError):
        _ = Meteoroid(x=0,
                        y=0,
                        z=100e3,
                        vx=-20e3 * np.cos(45 * np.pi / 180),
                        vy=[0, 1, 2, 3],
                        vz=-20e3 * np.sin(45 * np.pi / 180),
                        radius=150,
                        mass=0.6e3 * (4 * np.pi / 3) * (150 ** 3),
                        sigma=1e4,
                        rho=0.6e3,
                        eta=2.5e6)

    with pytest.raises(TypeError):
        _ = Meteoroid(x=0,
                        y=0,
                        z=0,
                        vx=-20e3 * np.cos(45 * np.pi / 180),
                        vy=0,
                        vz=[0, 1, 2, 3],
                        radius=150,
                        mass=0.6e3 * (4 * np.pi / 3) * (150 ** 3),
                        sigma=1e4,
                        rho=0.6e3,
                        eta=2.5e6)
