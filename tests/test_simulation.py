# pylint: disable=duplicate-code

"""
Module docstring:
"""

import pytest
import numpy as np
from atmosentry.meteoroid import Meteoroid
from atmosentry import Simulation


def test_simulation():
    """
    Docstring
    """

    sim = Simulation()
    assert sim
    assert sim.t == 0
    assert sim.Cd == 0.7
    assert sim.Ch == 0.02
    assert sim.Mpl == 5.97e24
    assert sim.Rpl == 6371e3
    assert sim.rho0 == 1.225
    assert sim.H == 7.2e3
    assert sim.Nfrag == 2
    assert sim.fragments_track is True


def test_simulation_impactor():
    """
    Docstring
    """

    impactor = Meteoroid(x=0,
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

    sim = Simulation()
    sim.impactor = impactor

    assert sim.impactor == impactor


@pytest.mark.parametrize("invalid_value", [0])
def test_simulation_setters(invalid_value):
    """
    Docstring
    """

    sim = Simulation()

    with pytest.raises(TypeError):
        sim.t = invalid_value
        sim.Cd = invalid_value
        sim.Ch = invalid_value
        sim.Mpl = invalid_value
        sim.Rpl = invalid_value
        sim.H = invalid_value
        sim.rho0 = invalid_value
        sim.impactor = invalid_value


@pytest.mark.parametrize("invalid_value", [0, 'string', [1, 2, 3]])
def test_simulation_setters_bool(invalid_value):
    """
    Docstring
    """

    sim = Simulation()

    with pytest.raises(TypeError):
        sim.fragments_track = invalid_value


def test_simulation_setters_fragments():
    """
    Docstring
    """

    sim = Simulation()

    impactor = Meteoroid(x=0,
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
        sim.fragments = impactor
        sim.fragments = [impactor]
        sim.fragments = [1, 2, 3]
        sim.fragments = np.array([1, 2, 3])
        sim.fragments = np.array([1, 2, 3, impactor])
        sim.fragments = 'impactor'
