"""
Add docstring...
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from atmosentry.meteoroid import Meteoroid
from integrator import run
from fragments import generate_fragments


class Simulation():
    """
    This is the atmospheric entry simulation class.
    """

    def __init__(self, 
                 t=0.,
                 fragments=np.array([]),
                 Cd=0.7,
                 Ch=0.02,
                 Cl=0.001,
                 Mpl=5.97e24,
                 Rpl=6371e3,
                 rho0=1.225,
                 H=7.2e3,
                 alpha=0.25,
                 beta=0.5,
                 Nfrag=2
                 ):
        
        self.t = t
        self.fragments = fragments
        self.Cd=Cd
        self.Ch=Ch
        self.Cl=Cl
        self.Mpl=Mpl
        self.Rpl=Rpl
        self.rho0 = rho0
        self.H=H
        self.alpha = alpha
        self.beta = beta
        self.Nfrag = Nfrag

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
    def Cl(self):
        """
        Docstring

        Returns:
            C_l: ADD DESCRIPTION
        """
        return self._Cl

    @Cl.setter
    def Cl(self, Cl):
        """
        Docstring

        Args:
            C_l (float): ADD DESCRIPTION

        Raises:
            TypeError: ADD DESCRIPTION
        """
        if not isinstance(Cl, float):
            raise TypeError("Heat transfer coefficient must be a float.")
        if isinstance(Cl, float):
            self._Cl = Cl

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
                         theta: float, 
                         radius: float, 
                         dEkindh: float, 
                         x: float, 
                         y: float, 
                         z: float, 
                         vx: float, 
                         vy: float, 
                         vz: float):
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
        impactor.theta = theta
        impactor.dEkindh = dEkindh
    
    def integrate(self):

        t, mass, theta, radius, dEkindh, x, y, z, vx, vy, vz =\
            run(self._impactor,
                        self._Cd,
                        self._Ch,
                        self._Cl,
                        self._Rpl,
                        self._Mpl,
                        self._rho0,
                        self._H,
                        N_c=2.
            )
    
        sim._t = t
        self.update_meteoroid(self._impactor, mass, theta, radius, dEkindh, x, y, z, vx, vy, vz)

        if self._impactor.z[-1] > 1:

            child_fragments = generate_fragments(self._impactor, self.rho0, self.H, self.alpha, self.beta, self.Nfrag)
            
            while len(child_fragments) > 0:
                fragments_tmp = []

                for fragment in child_fragments:
                    print(fragment)

                    _, mass_f, theta_f, radius_f, dEkindh_f, x_f, y_f, z_f, vx_f, vy_f, vz_f =\
                        run(fragment,
                            self._Cd,
                            self._Ch,
                            self._Cl,
                            self._Rpl,
                            self._Mpl,
                            self._rho0,
                            self._H,
                            N_c=2.
                        )
                    self.update_meteoroid(fragment, mass_f, theta_f, radius_f, dEkindh_f, x_f, y_f, z_f, vx_f, vy_f, vz_f)
                    
                    if z_f[-1] < 1:
                        self._fragments = np.append(self._fragments, fragment)
                    elif mass_f[-1] < 0.01 * self._impactor.mass[0]:
                        self._fragments = np.append(self._fragments, fragment)
                    else:
                        self._fragments = np.append(self._fragments, fragment)
                        child_frags = generate_fragments(fragment, self.rho0, self.H, self.alpha, self.beta, self.Nfrag)
                        fragments_tmp = np.append(fragments_tmp, child_frags)
                
                child_fragments = fragments_tmp

rho_com = 0.6e3
rho_atm0 = 1.225

theta0 = 45. * np.pi / 180.
V0 = 20e3
R0 = 150

impactor = Meteoroid(x=0,
                     y=0,
                     z=100e3,
                     vx=-V0 * np.cos(theta0),
                     vy=0,
                     vz=-V0 * np.sin(theta0),
                     theta=theta0,
                     radius=R0,
                     mass=rho_com * (4 * np.pi / 3) * (R0 ** 3),
                     sigma=1e4,
                     rho=0.6e3,
                     eta=2.5e6)

sim = Simulation()

sim.impactor = impactor

sim.integrate()

vel = np.sqrt(sim.impactor.vx ** 2 + sim.impactor.vy ** 2 + sim.impactor.vz ** 2)

plt.plot(vel / 1e3, sim.impactor.z / 1e3)
for fragment in sim.fragments:
        
    vel = np.sqrt(fragment.vx ** 2 + fragment.vy ** 2 + fragment.vz ** 2)

    plt.plot(vel / 1e3, fragment.z / 1e3, c='k')
    plt.plot(vel[0] / 1e3, fragment.z[0] / 1e3, 'x', c='k')

plt.ylim(0, 100)

plt.minorticks_on()

plt.show()

