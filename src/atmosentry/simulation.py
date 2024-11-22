"""
Add docstring...
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from atmosentry.meteoroid import Meteoroid
from .integrator import run
from .fragments import generate_fragments


class Simulation():
    """
    This is the atmospheric entry simulation class.
    """

    def __init__(self, 
                 t=0.,
                 fragments=np.array([]),
                 Cd=0.7,
                 Ch=0.02,
                 Mpl=5.97e24,
                 Rpl=6371e3,
                 rho0=1.225,
                 H=7.2e3,
                 Nfrag=2,
                 fragments_track=True
                 ):
        
        self.t = t
        self.fragments = fragments
        self.Cd=Cd
        self.Ch=Ch
        self.Mpl=Mpl
        self.Rpl=Rpl
        self.rho0 = rho0
        self.H=H
        self.Nfrag = Nfrag
        self.fragments_track = fragments_track

    @property
    def t(self):
        """
        Docstring

        Returns:
            t: ADD DESCRIPTION
        """
        return self._t

    @t.setter
    def t(self, t):
        """
        Docstring

        Args:
            t (float): ADD DESCRIPTION

        Raises:
            TypeError: ADD DESCRIPTION
        """
        if not isinstance(t, float):
            raise TypeError("Simulation time must be a float.")
        if isinstance(t, float):
            self._t = t

    @property
    def fragments(self):
        """
        Docstring

        Returns:
            fragments: ADD DESCRIPTION
        """
        return self._fragments
    
    @fragments.setter
    def fragments(self, fragments):
        """
        Docstring

        Args:
            fragments (float): ADD DESCRIPTION

        Raises:
            TypeError: ADD DESCRIPTION
        """
        if not isinstance(fragments, np.ndarray):
            raise TypeError("Must store fragments as an array.")
        for fragment in fragments:
            if not isinstance(fragment, Meteoroid):
                raise TypeError("Not a Meteoroid object.")
        if isinstance(fragments, np.ndarray):
            self._fragments = fragments

    @property
    def Cd(self):
        """
        Docstring

        Returns:
            Cd: ADD DESCRIPTION
        """
        return self._Cd

    @Cd.setter
    def Cd(self, Cd):
        """
        Docstring

        Args:
            C_d (float): ADD DESCRIPTION

        Raises:
            TypeError: ADD DESCRIPTION
        """
        if not isinstance(Cd, float):
            raise TypeError("Drag coefficient must be a float.")
        if isinstance(Cd, float):
            self._Cd = Cd

    @property
    def Ch(self):
        """
        Docstring

        Returns:
            C_h: ADD DESCRIPTION
        """
        return self._Ch

    @Ch.setter
    def Ch(self, Ch):
        """
        Docstring

        Args:
            C_h (float): ADD DESCRIPTION

        Raises:
            TypeError: ADD DESCRIPTION
        """
        if not isinstance(Ch, float):
            raise TypeError("Heat transfer coefficient must be a float.")
        if isinstance(Ch, float):
            self._Ch = Ch

    @property
    def Mpl(self):
        """
        Docstring

        Returns:
            M_pl: ADD DESCRIPTION
        """
        return self._Mpl

    @Mpl.setter
    def Mpl(self, Mpl):
        """
        Docstring

        Args:
            M_pl (float): ADD DESCRIPTION

        Raises:
            TypeError: ADD DESCRIPTION
        """
        if not isinstance(Mpl, float):
            raise TypeError("Planet mass must be a float.")
        if isinstance(Mpl, float):
            self._Mpl = Mpl

    @property
    def Rpl(self):
        """
        Docstring

        Returns:
            R_pl: ADD DESCRIPTION
        """
        return self._Rpl

    @Rpl.setter
    def Rpl(self, Rpl):
        """
        Docstring

        Args:
            R_pl (float): ADD DESCRIPTION

        Raises:
            TypeError: ADD DESCRIPTION
        """
        if not isinstance(Rpl, float):
            raise TypeError("Planet radius must be a float.")
        if isinstance(Rpl, float):
            self._Rpl = Rpl

    @property
    def H(self):
        """
        Docstring

        Returns:
            H: ADD DESCRIPTION
        """
        return self._H

    @H.setter
    def H(self, H):
        """
        Docstring

        Args:
            H (float): ADD DESCRIPTION

        Raises:
            TypeError: ADD DESCRIPTION
        """
        if not isinstance(H, float):
            raise TypeError("Atmospheric scale height must be a float.")
        if isinstance(H, float):
            self._H = H

    @property
    def fragments_track(self):
        """
        Docstring

        Returns:
            fragments_track: ADD DESCRIPTION
        """
        return self._fragments_track

    @fragments_track.setter
    def fragments_track(self, fragments_track):
        """
        Docstring

        Args:
            fragments_track (float): ADD DESCRIPTION

        Raises:
            TypeError: ADD DESCRIPTION
        """
        if not isinstance(fragments_track, bool):
            raise TypeError("Fragment tracking must be a bool.")
        if isinstance(fragments_track, bool):
            self._fragments_track = fragments_track

    @property
    def rho0(self):
        """
        Docstring

        Returns:
            rho0: ADD DESCRIPTION
        """
        return self._rho0

    @rho0.setter
    def rho0(self, rho0):
        """
        Docstring

        Args:
            rho0 (float): ADD DESCRIPTION

        Raises:
            TypeError: ADD DESCRIPTION
        """
        if not isinstance(rho0, float):
            raise TypeError("Atmospheric surface density must be a float.")
        if isinstance(rho0, float):
            self._rho0 = rho0

    @property
    def impactor(self):
        """
        Docstring
        """
        return self._impactor
    
    @impactor.setter
    def impactor(self, value):
        """
        Docstring
        """
        if not isinstance(value, Meteoroid):
            raise TypeError("Impactor must be a Meteoroid object.")
        if isinstance(value, Meteoroid):
            self._impactor = value
        
    @staticmethod
    def update_meteoroid(impactor: Meteoroid, 
                         mass: float, 
                         radius: float, 
                         dM: float,
                         dEkin: float,
                         x: float, 
                         y: float, 
                         z: float, 
                         vx: float, 
                         vy: float, 
                         vz: float,
                         t: list):
        """
        Docstring
        """

        impactor.x = x
        impactor.y = y
        impactor.z = z
        impactor.vx = vx
        impactor.vy = vy
        impactor.vz = vz
        impactor.mass = mass
        impactor.radius = radius
        impactor.dM = dM
        impactor.dEkin = dEkin
        impactor.t = t + impactor.t_init
    
    def integrate(self):

        t, mass, radius, dM, dEkin, x, y, z, vx, vy, vz, _ =\
            run(self._impactor,
                        self._Cd,
                        self._Ch,
                        self._Rpl,
                        self._Mpl,
                        self._rho0,
                        self._H,
                        N_c=2.
            )
    
        self._t = t
        self.update_meteoroid(self._impactor, mass, radius, dM, dEkin, x, y, z, vx, vy, vz, t)

        if self._fragments_track:
            if self._impactor.z[-1] > 1:
                self._impactor.children = True

                child_fragments = generate_fragments(self._impactor, self.rho0, self.H, self.Nfrag)
                
                while len(child_fragments) > 0:
                    fragments_tmp = []

                    for fragment in child_fragments:
                        print(f'R_0={self._impactor.radius[0]}m, N_frags={len(child_fragments)}', end='\r', flush=True)

                        t_f, mass_f, radius_f, dM_f, dEkin_f, x_f, y_f, z_f, vx_f, vy_f, vz_f, N_RTf =\
                            run(fragment,
                                self._Cd,
                                self._Ch,
                                self._Rpl,
                                self._Mpl,
                                self._rho0,
                                self._H,
                                N_c=2.
                            )
                        self.update_meteoroid(fragment, mass_f, radius_f, dM_f, dEkin_f, x_f, y_f, z_f, vx_f, vy_f, vz_f, t_f)
                        
                        if z_f[-1] < 1:
                            self._fragments = np.append(self._fragments, fragment)
                        elif mass_f[-1] < 0.005 * self._impactor.mass[0]:
                            mass_f = np.append(mass_f, mass_f[-1])
                            radius_f = np.append(radius_f, radius_f[-1])
                            x_f = np.append(x_f, x_f[-1])
                            y_f = np.append(y_f, y_f[-1])
                            z_f = np.append(z_f, z_f[-1])
                            vx_f = np.append(vx_f, vx_f[-1])
                            vy_f = np.append(vy_f, vy_f[-1])
                            vz_f = np.append(vz_f, vz_f[-1])

                            vel = vx_f[-1] ** 2 + vy_f[-1] ** 2 + vz_f[-1] ** 2

                            dM_f = np.append(dM_f, mass_f[-1])
                            dEkin_f = np.append(dEkin_f, 0.5 * mass_f[-1] * vel)

                            self.update_meteoroid(fragment, mass_f, radius_f, dM_f, dEkin_f, x_f, y_f, z_f, vx_f, vy_f, vz_f, t_f)
                            self._fragments = np.append(self._fragments, fragment)

                        else:
                            fragment.children = True
                            self._fragments = np.append(self._fragments, fragment)
                            child_frags = generate_fragments(fragment, self.rho0, self.H, self.Nfrag)
                            fragments_tmp = np.append(fragments_tmp, child_frags)
                    
                    child_fragments = fragments_tmp
