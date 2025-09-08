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

class LorentzBoost:
    """
    Implements a general Lorentz boost along an arbitrary direction.
    
    A Lorentz boost transforms coordinates and momenta from one inertial reference 
    frame to another moving at a constant velocity `beta_vec` relative to the first.
    """
    
    def __init__(self, beta_vec: np.ndarray):
        """
        Initialize the Lorentz boost.
        
        Args:
            beta_vec (np.ndarray): A 3D numpy array representing the boost velocity vector (v/c).
                Boosting by `beta_vec` goes to a frame moving at velocity `beta_vec`.
                To go to the rest frame of a particle, boost by `-particle.beta`.
        """
        self.beta_vec = np.asarray(beta_vec, dtype=float)
        b2 = np.sum(self.beta_vec**2)
        if b2 >= 1.0:
            raise ValueError(f"Magnitude of beta vector must be < 1. Got beta^2 = {b2}")
        self.beta_sq = b2

    @property
    def gamma(self) -> float:
        """Lorentz factor computed from the magnitude of the boost velocity."""
        if self.beta_sq == 0:
            return 1.0
        return 1.0 / np.sqrt(1.0 - self.beta_sq)

    def boost_matrix(self) -> np.ndarray:
        """
        Returns the explicit 4x4 general Lorentz boost matrix.
        
        For a boost vector beta = (bx, by, bz), the matrix Lambda transforms a 
        4-vector V via matrix multiplication: V' = Lambda @ V.
        """
        b2 = self.beta_sq
        g = self.gamma
        
        if b2 == 0:
            return np.eye(4)
            
        bx, by, bz = self.beta_vec
        
        # The gamma factor scaled for the spatial components
        factor = (g - 1.0) / b2
        
        # Construct the 4x4 matrix
        Lambda = np.array([
            [g, -g*bx, -g*by, -g*bz],
            [-g*bx, 1 + factor*bx**2, factor*bx*by, factor*bx*bz],
            [-g*by, factor*by*bx, 1 + factor*by**2, factor*by*bz],
            [-g*bz, factor*bz*bx, factor*bz*by, 1 + factor*bz**2]
        ])
        return Lambda

    def boost(self, fourvector: FourVector) -> FourVector:
        """
        Apply the Lorentz boost to a single FourVector.
        
        Physical concept:
            This returns the energy and momentum of the particle as measured by an 
            observer in the new reference frame. Note that transverse momentum 
            (perpendicular to the boost) is unchanged, while parallel momentum 
            and energy mix.
            
        Args:
            fourvector: The original FourVector.
        Returns:
            The Lorentz-boosted FourVector.
        """
        Lambda = self.boost_matrix()
        v_prime = Lambda @ fourvector.to_array()
        return FourVector(*v_prime)

    def boost_many(self, array_of_fourvectors: np.ndarray) -> np.ndarray:
        """
        Vectorized numpy implementation for boosting N particles simultaneously.
        
        Args:
            array_of_fourvectors: NumPy array of shape (N, 4) where each row is [E, px, py, pz].
        Returns:
            NumPy array of shape (N, 4) with the boosted components.
        """
        Lambda = self.boost_matrix()
        # Lambda is (4,4), array is (N,4). We want (N,4) result.
        # array_of_fourvectors.T is (4,N).
        # Lambda @ (4,N) -> (4,N)
        # Transpose back to (N,4)
        return (Lambda @ array_of_fourvectors.T).T

