# pylint: disable=C0103
# pylint: disable=W0102
# pylint: disable=R0903

"""
Module: meteoroid

This module defines the Meteoroid class, which contains the physical properties, 
and dynamic state of a meteoroid entering a planetary atmosphere. 

At some point should come back and check the correct types in setters/getters....

I THINK THERE WILL ALSO BE A PROBLEM HERE: WE SHOULD NOT ALLOW USERS TO SET ARRAYS AS X,Y,Z 
ETC. - BUT THIS IS SOMETHING THE CODE DOES DO....
"""
import numpy as np


class MeteoroidState():
    """
    Docstring
    """

    def __init__(self,
                 x=None,
                 y=None,
                 z=None,
                 vx=None,
                 vy=None,
                 vz=None,
                 radius=None,
                 mass=None,
                 dM=None,
                 dEkin=None,
                 t=None
                 ):
        """
        Docstring - NEED TO ADD GETTERS AT SOME POINT SOON!
        """

        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.radius = radius
        self.mass = mass
        self.dM = dM
        self.dEkin = dEkin
        self.t = t

    def update(self,
               x,
               y,
               z,
               vx,
               vy,
               vz,
               radius,
               mass,
               dM,
               dEkin,
               t):
        """
        Docstring
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"x must be of type np.ndarray, not {type(x).__name__}")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y must be of type np.ndarray, not {type(y).__name__}")
        if not isinstance(z, np.ndarray):
            raise TypeError(f"z must be of type np.ndarray, not {type(z).__name__}")
        if not isinstance(vx, np.ndarray):
            raise TypeError(f"vx must be of type np.ndarray, not {type(vx).__name__}")
        if not isinstance(vy, np.ndarray):
            raise TypeError(f"vy must be of type np.ndarray, not {type(vy).__name__}")
        if not isinstance(vz, np.ndarray):
            raise TypeError(f"vz must be of type np.ndarray, not {type(vz).__name__}")
        if not isinstance(radius, np.ndarray):
            raise TypeError(f"radius must be of type np.ndarray, not {type(radius).__name__}")
        if not isinstance(mass, np.ndarray):
            raise TypeError(f"mass must be of type np.ndarray, not {type(mass).__name__}")
        if not isinstance(dM, np.ndarray):
            raise TypeError(f"dM must be of type np.ndarray, not {type(dM).__name__}")
        if not isinstance(dEkin, np.ndarray):
            raise TypeError(f"dEkin must be of type np.ndarray, not {type(dEkin).__name__}")
        if not isinstance(t, np.ndarray):
            raise TypeError(f"t must be of type np.ndarray, not {type(t).__name__}")

        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.radius = radius
        self.mass = mass
        self.dM = dM
        self.dEkin = dEkin
        self.t = t


class Meteoroid():
    """
    Describes the physical properties, and trajectory of a meteoroid during atmospheric
    entry. The object also tracks whether the meteoroid has generated child fragments.

    Attributes:
    -----------
        x (float): The x-position of the meteoroid [m]
        y (float): The y-position of the meteoroid [m]
        z (float): The z-position of the meteoroid [m]
        vx (float): The x-component of the meteoroid's velocity [m/s] 
        vy (float): The y-component of the meteoroid's velocity [m/s] 
        vz (float): The z-component of the meteoroid's velocity [m/s] 
        radius (float): Radius of the meteoroid [m]]
        mass (float): Mass of the meteoroid [kg]]
        sigma (float): Tensile strength of the meteoroid [Pa]
        rho (float): Bulk density of the meteoroid [kg/m^3]
        eta (float): Heat of ablation of the meteoroid [J/kg]
        dM (float, optional): The mass loss of the meteoroid [kg/s]
        dEkin (float, optional): The energy loss of the meteoroid [J/s]
        children (bool, optional): Meteoroid's child fragments (default: False)
        t_init (float, optional): Initial simulation time [s] (default: 0)
        t (list, optional): Simulation time

    """

    def __init__(self,
                 x,
                 y,
                 z,
                 vx,
                 vy,
                 vz,
                 radius,
                 mass,
                 sigma,
                 rho,
                 eta,
                 children=False,
                 t_init=0,
                 state=None):
        """
        Initializes a Meteoroid object with the given physical properties and initial state.

        Args:
        -----
            x (float): Initial x-coordinate of the meteoroid [m]
            y (float): Initial y-coordinate of the meteoroid [m]
            z (float): Initial z-coordinate (altitude) of the meteoroid [m]
            vx (float): Initial velocity in the x-direction [m/s]
            vy (float): Initial velocity in the y-direction [m/s]
            vz (float): Initial velocity in the z-direction [m/s]
            radius (float): Radius of the meteoroid [m]]
            mass (float): Mass of the meteoroid in [kg]
            sigma (float): Tensile strength of the meteoroid [Pa]
            rho (float): Bulk density of the meteoroid [kg/m^3]]
            eta (float): Heat of ablation of the meteoroid [J/kg]
            dM (float, optional): The mass loss of the meteoroid [kg/s]
            dEkin (float, optional): The energy loss of the meteoroid [J/s]
            children (bool, optional): Meteoroid's child fragments (default: False)
            t_init (float, optional): Initial simulation time [s] (default: 0)
            t (list, optional): Simulation time

        """

        if state is None:
            state = MeteoroidState()

        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.radius = radius
        self.mass = mass
        self.sigma = sigma
        self.rho = rho
        self.eta = eta
        self.children = children
        self.t_init = t_init
        self.state = state

    @property
    def x(self):
        """Getter for the meteoroid's x-coordinate."""
        return self._x

    @x.setter
    def x(self, value):
        """Setter for the meteoroid's x-coordinate."""

        if not isinstance(value, (float, int, np.ndarray)):
            raise TypeError()
        self._x = value

    @property
    def y(self):
        """Getter for the meteoroid's y-coordinate."""
        return self._y

    @y.setter
    def y(self, value):
        """Setter for the meteoroid's y-coordinate."""

        if not isinstance(value, (float, int, np.ndarray)):
            raise TypeError()
        self._y = value

    @property
    def z(self):
        """Getter for the meteoroid's z-coordinate."""
        return self._z

    @z.setter
    def z(self, value):
        """Setter for the meteoroid's z-coordinate."""

        if not isinstance(value, (float, int, np.ndarray)):
            raise TypeError()
        self._z = value

    @property
    def vx(self):
        """Getter for the meteoroid's velocity in the x-direction."""
        return self._vx

    @vx.setter
    def vx(self, value):
        """Setter for the meteoroid's velocity in the x-direction."""

        if not isinstance(value, (float, int, np.ndarray)):
            raise TypeError()
        self._vx = value

    @property
    def vy(self):
        """Getter for the meteoroid's velocity in the y-direction."""
        return self._vy

    @vy.setter
    def vy(self, value):
        """Setter for the meteoroid's velocity in the y-direction."""

        if not isinstance(value, (float, int, np.ndarray)):
            raise TypeError()
        self._vy = value

    @property
    def vz(self):
        """Getter for the meteoroid's velocity in the z-direction."""
        return self._vz

    @vz.setter
    def vz(self, value):
        """Setter for the meteoroid's velocity in the z-direction."""

        if not isinstance(value, (float, int, np.ndarray)):
            raise TypeError()
        self._vz = value

    @property
    def radius(self):
        """Getter for the meteoroid's radius."""
        return self._radius

    @radius.setter
    def radius(self, value):
        """Setter for the meteoroid's radius."""
        self._radius = value

    @property
    def mass(self):
        """Getter for the meteoroid's mass."""
        return self._mass

    @mass.setter
    def mass(self, value):
        """Setter for the meteoroid's mass."""
        self._mass = value

    @property
    def sigma(self):
        """Getter for the meteoroid's tensile strength."""
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        """Setter for the meteoroid's tensile strength."""
        self._sigma = value

    @property
    def rho(self):
        """Getter for the meteoroid's bulk density."""
        return self._rho

    @rho.setter
    def rho(self, value):
        """Setter for the meteoroid's bulk density."""
        self._rho = value

    @property
    def eta(self):
        """Getter for the meteoroid's heat of ablation."""
        return self._eta

    @eta.setter
    def eta(self, value):
        """Setter for the meteoroid's heat of ablation."""
        self._eta = value

    @property
    def children(self):
        """Getter for the meteoroid's children fragments status."""
        return self._children

    @children.setter
    def children(self, value):
        """Setter for the meteoroid's children fragments status."""
        self._children = value

    @property
    def t_init(self):
        """Getter for the meteoroid's initial time."""
        return self._t_init

    @t_init.setter
    def t_init(self, value):
        """Setter for the meteoroid's initial time."""
        self._t_init = value
