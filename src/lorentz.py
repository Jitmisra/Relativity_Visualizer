import numpy as np
from typing import List, Dict, Union

# Natural units: c = 1
C_LIGHT = 1.0

class FourVector:
    """
    Represents a 4-vector (E, px, py, pz) in natural units where c=1.
    
    In special relativity, space and time are fundamentally intertwined into
    a single 4-dimensional continuous spacetime called Minkowski space. 
    A 4-vector is a mathematical object that transforms in a specific way under
    changes of reference frames (Lorentz transformations).
    
    The components (E, px, py, pz) represent the energy and 3-momentum of a 
    particle. Since c=1, energy has the same units as momentum.
    """
    
    def __init__(self, E: float, px: float, py: float, pz: float):
        """
        Initialize the 4-vector components.
        
        Args:
            E (float): Energy (timelike component).
            px (float): x momentum (spacelike component).
            py (float): y momentum (spacelike component).
            pz (float): z momentum (spacelike component).
        """
        self.E = float(E)
        self.px = float(px)
        self.py = float(py)
        self.pz = float(pz)
        self._vec = np.array([self.E, self.px, self.py, self.pz])

    @property
    def magnitude_squared(self) -> float:
        """
        Calculates the invariant magnitude squared of the 4-vector: 
        m^2 = E^2 - |p|^2.
        
        This is a Lorentz scalar, meaning it is exactly the same in all reference frames.
        Physically, this represents the rest mass (squared) of the particle.
        """
        p2 = self.px**2 + self.py**2 + self.pz**2
        return self.E**2 - p2

    @property
    def beta(self) -> np.ndarray:
        """
        The relativistic velocity vector, beta = p_vec / E.
        
        This represents the 3-velocity of the particle relative to the speed of light.
        Since we use units where c=1, |beta| must be strictly less than 1 for massive 
        particles, and exactly 1 for massless particles like photons.
        """
        if self.E == 0:
            return np.zeros(3)
        return np.array([self.px, self.py, self.pz]) / self.E

    @property
    def gamma(self) -> float:
        """
        The Lorentz factor, gamma = 1 / sqrt(1 - |beta|^2).
        
        It quantifies the magnitude of relativistic effects like time dilation 
        and length contraction. As beta approaches 1 (the speed of light), 
        gamma tends to infinity.
        """
        b2 = np.sum(self.beta**2)
        if b2 >= 1.0:
            return float('inf')
        return 1.0 / np.sqrt(1.0 - b2)

    def __add__(self, other: 'FourVector') -> 'FourVector':
        """
        Adds two 4-vectors component-wise.
        
        Physically, this corresponds to finding the total 4-momentum of a system
        of particles. The beauty of 4-vectors is that the sum of multiple valid 
        4-vectors is itself a valid 4-vector.
        """
        return FourVector(
            self.E + other.E,
            self.px + other.px,
            self.py + other.py,
            self.pz + other.pz
        )

    def __repr__(self) -> str:
        return f"FourVector(E={self.E:.4f}, px={self.px:.4f}, py={self.py:.4f}, pz={self.pz:.4f})"
        
    def to_array(self) -> np.ndarray:
        """Returns the 4-vector as a 1D numpy array [E, px, py, pz]."""
        return self._vec.copy()

    @classmethod
    def from_pT_rapidity_phi(cls, pT: float, y: float, phi: float, m: float = 0.0) -> 'FourVector':
        """
        Constructs a FourVector from cylindrical coordinates used in particle physics 
        experiments (like ATLAS or CMS at CERN).
        
        Args:
            pT (float): Transverse momentum (momentum perpendicular to the beam axis).
            y (float): Rapidity (a relativistic measure of longitudinal velocity).
            phi (float): Azimuthal angle around the beam axis.
            m (float, optional): Invariant mass of the particle. Defaults to 0.0.
            
        Physical explanation:
            The beam axis is typically taken as the z-axis. Rapidity `y` is preferred 
            over velocity or standard polar angle because differences in rapidity are 
            invariant under Lorentz boosts along the z-axis.
        """
        px = pT * np.cos(phi)
        py = pT * np.sin(phi)
        pz = pT * np.sinh(y)
        
        if m == 0.0:
            E = pT * np.cosh(y)
        else:
            p_mag_sq = px**2 + py**2 + pz**2
            E = np.sqrt(p_mag_sq + m**2)
            
        return cls(E, px, py, pz)

