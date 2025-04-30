from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from enum import Enum

class MorphicRole(Enum):
    COMPUTE = 0  # C - operator/function aspect
    VALUE = 1    # V - state/value aspect  
    TYPE = 2     # T - type/structural aspect

@dataclass
class BitMorphology:
    """
    Core structure for bit-morphology that functions as both data and process.
    Organized as CVVV (bra) | TTTT (ket) for an 8-bit morphic unit.
    """
    raw: int  # Raw 8-bit value
    
    @property
    def bra(self) -> int:
        """Extract the CVVV components (top-nibble)"""
        return (self.raw >> 4) & 0xF
    
    @property
    def ket(self) -> int:
        """Extract the TTTT components (bottom-nibble)"""
        return self.raw & 0xF
    
    @property
    def compute(self) -> bool:
        """Extract the C component (operator nature)"""
        return bool((self.raw >> 7) & 0x1)
    
    @property
    def value_triplet(self) -> Tuple[bool, bool, bool]:
        """Extract the VVV components (state values)"""
        return (
            bool((self.raw >> 6) & 0x1),
            bool((self.raw >> 5) & 0x1),
            bool((self.raw >> 4) & 0x1)
        )
    
    @property
    def type_quartet(self) -> Tuple[bool, bool, bool, bool]:
        """Extract the TTTT components (type structure)"""
        return (
            bool((self.raw >> 3) & 0x1),
            bool((self.raw >> 2) & 0x1),
            bool((self.raw >> 1) & 0x1),
            bool(self.raw & 0x1)
        )
    
    def morph(self, other: 'BitMorphology') -> 'BitMorphology':
        """
        Apply this morphology to another, creating a transformation.
        The compute bit determines whether this is a forward or adjoint operation.
        """
        if self.compute:
            # Forward morphism - bra transforms ket
            result = (self.bra << 4) | other.ket
        else:
            # Adjoint morphism - ket transforms bra
            result = (other.bra << 4) | self.ket
            
        return BitMorphology(result)
    
    def inner_product(self, other: 'BitMorphology') -> int:
        """
        Compute the inner product as a measure of morphic compatibility.
        Higher values indicate greater resonance between morphologies.
        """
        # XOR the bras (difference = 0 means compatibility)
        bra_diff = self.bra ^ other.bra
        # XOR the kets (difference = 0 means compatibility)
        ket_diff = self.ket ^ other.ket
        
        # Count the number of matching bits (8 - number of differing bits)
        compatibility = 8 - bin(bra_diff).count('1') - bin(ket_diff).count('1')
        return compatibility
    
    def evolve(self) -> 'BitMorphology':
        """
        Autonomous evolution rule for this cellular automaton unit.
        Follows a morphic field principle based on internal structure.
        """
        # Example evolution rule: rotate values by compute bit
        if self.compute:
            # Rotate right - more "entropic" behavior
            new_raw = ((self.raw & 0x1) << 7) | (self.raw >> 1)
        else:
            # Rotate left - more "negentropic" behavior
            new_raw = ((self.raw << 1) & 0xFF) | ((self.raw >> 7) & 0x1)
            
        return BitMorphology(new_raw)
    
    @staticmethod
    def create_from_components(
        compute: bool,
        values: Tuple[bool, bool, bool],
        types: Tuple[bool, bool, bool, bool]
    ) -> 'BitMorphology':
        """Create a BitMorphology from component parts"""
        raw = (int(compute) << 7)
        raw |= (int(values[0]) << 6)
        raw |= (int(values[1]) << 5)
        raw |= (int(values[2]) << 4)
        raw |= (int(types[0]) << 3)
        raw |= (int(types[1]) << 2)
        raw |= (int(types[2]) << 1)
        raw |= int(types[3])
        return BitMorphology(raw)
    
    def __repr__(self) -> str:
        """Human-readable representation of the morphology"""
        c = "C" if self.compute else "c"
        v = "".join(["V" if v else "v" for v in self.value_triplet])
        t = "".join(["T" if t else "t" for t in self.type_quartet])
        return f"{c}{v}|{t} [0x{self.raw:02x}]"


class MorphicField:
    """
    A lattice of BitMorphology entities that evolve according to rulial dynamics.
    Functions as a distributed quantum-like computational fabric.
    """
    def __init__(self, size: Tuple[int, int]):
        self.width, self.height = size
        self.lattice = np.zeros((self.height, self.width), dtype=np.uint8)
        self.morphologies = [[BitMorphology(0) for _ in range(self.width)] 
                             for _ in range(self.height)]
        
    def initialize_random(self, seed: Optional[int] = None) -> None:
        """Initialize the field with random values"""
        rng = np.random.RandomState(seed)
        self.lattice = rng.randint(0, 256, (self.height, self.width), dtype=np.uint8)
        
        # Update the morphologies to match the raw values
        for y in range(self.height):
            for x in range(self.width):
                self.morphologies[y][x] = BitMorphology(self.lattice[y, x])
    
    def initialize_kernel(self, kernel_type: str) -> None:
        """Initialize with a specific kernel pattern"""
        if kernel_type == "gaussian_white":
            # Gaussian white noise centered around 128
            self.lattice = np.random.normal(128, 40, (self.height, self.width))
            self.lattice = np.clip(self.lattice, 0, 255).astype(np.uint8)
        elif kernel_type == "quine":
            # Self-replicating pattern - simple example
            # Creates a pattern that tends to recreate itself under the evolution rules
            midpoint = (self.width // 2, self.height // 2)
            for y in range(self.height):
                for x in range(self.width):
                    # Distance from center
                    dist = ((x - midpoint[0])**2 + (y - midpoint[1])**2)**0.5
                    # Create a radial pattern with alternating compute bits
                    compute = (int(dist) % 2 == 0)
                    values = (dist % 3 < 1, dist % 5 < 2, dist % 7 < 3)
                    types = (dist % 2 < 1, dist % 3 < 1, dist % 5 < 2, dist % 7 < 3)
                    
                    morph = BitMorphology.create_from_components(compute, values, types)
                    self.lattice[y, x] = morph.raw
                    self.morphologies[y][x] = morph
        
        # Update the morphologies to match the raw values
        for y in range(self.height):
            for x in range(self.width):
                self.morphologies[y][x] = BitMorphology(self.lattice[y, x])
    
    def step(self) -> None:
        """Evolve the entire field one step forward"""
        new_lattice = np.zeros_like(self.lattice)
        
        for y in range(self.height):
            for x in range(self.width):
                # Get the current morphology
                current = self.morphologies[y][x]
                
                # Get neighbors
                neighbors = self._get_neighbors(x, y)
                
                # Apply rulial dynamics
                new_morphology = self._apply_rulial_dynamics(current, neighbors)
                
                # Update the new lattice
                new_lattice[y, x] = new_morphology.raw
        
        # Update the lattice and morphologies
        self.lattice = new_lattice
        for y in range(self.height):
            for x in range(self.width):
                self.morphologies[y][x] = BitMorphology(self.lattice[y, x])
    
    def _get_neighbors(self, x: int, y: int) -> List[BitMorphology]:
        """Get the neighboring morphologies"""
        neighbors = []
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip the center
                
                nx, ny = (x + dx) % self.width, (y + dy) % self.height
                neighbors.append(self.morphologies[ny][nx])
        
        return neighbors
    
    def _apply_rulial_dynamics(self, center: BitMorphology, 
                              neighbors: List[BitMorphology]) -> BitMorphology:
        """
        Apply rulial dynamics based on the center and its neighbors.
        This is where the quantum-like probabilistic behavior emerges.
        """
        # First, find the neighbor with highest morphic resonance (inner product)
        max_resonance = -1
        most_resonant = None
        
        for neighbor in neighbors:
            resonance = center.inner_product(neighbor)
            if resonance > max_resonance:
                max_resonance = resonance
                most_resonant = neighbor
        
        # Apply different dynamics based on resonance levels
        if max_resonance >= 6:
            # High resonance - perform morphic transformation
            return center.morph(most_resonant)
        elif max_resonance >= 4:
            # Medium resonance - autonomous evolution
            return center.evolve()
        else:
            # Low resonance - maintain identity
            return BitMorphology(center.raw)
    
    def visualize(self) -> np.ndarray:
        """
        Create a visualization of the morphic field.
        Returns an RGB image representing the field state.
        """
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for y in range(self.height):
            for x in range(self.width):
                morph = self.morphologies[y][x]
                
                # Use different color channels for different aspects
                # Red channel: Compute bit
                image[y, x, 0] = 255 if morph.compute else 0
                
                # Green channel: Value bits
                v_sum = sum(int(v) for v in morph.value_triplet)
                image[y, x, 1] = int(v_sum * 255 / 3)
                
                # Blue channel: Type bits
                t_sum = sum(int(t) for t in morph.type_quartet)
                image[y, x, 2] = int(t_sum * 255 / 4)
        
        return image

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from enum import Enum

class MorphicRole(Enum):
    COMPUTE = 0  # C - operator/function aspect
    VALUE = 1    # V - state/value aspect  
    TYPE = 2     # T - type/structural aspect

@dataclass
class BitMorphology:
    """
    Core structure for bit-morphology that functions as both data and process.
    Organized as CVVV (bra) | TTTT (ket) for an 8-bit morphic unit.
    """
    raw: int  # Raw 8-bit value
    
    @property
    def bra(self) -> int:
        """Extract the CVVV components (top-nibble)"""
        return (self.raw >> 4) & 0xF
    
    @property
    def ket(self) -> int:
        """Extract the TTTT components (bottom-nibble)"""
        return self.raw & 0xF
    
    @property
    def compute(self) -> bool:
        """Extract the C component (operator nature)"""
        return bool((self.raw >> 7) & 0x1)
    
    @property
    def value_triplet(self) -> Tuple[bool, bool, bool]:
        """Extract the VVV components (state values)"""
        return (
            bool((self.raw >> 6) & 0x1),
            bool((self.raw >> 5) & 0x1),
            bool((self.raw >> 4) & 0x1)
        )
    
    @property
    def type_quartet(self) -> Tuple[bool, bool, bool, bool]:
        """Extract the TTTT components (type structure)"""
        return (
            bool((self.raw >> 3) & 0x1),
            bool((self.raw >> 2) & 0x1),
            bool((self.raw >> 1) & 0x1),
            bool(self.raw & 0x1)
        )
    
    def morph(self, other: 'BitMorphology') -> 'BitMorphology':
        """
        Apply this morphology to another, creating a transformation.
        The compute bit determines whether this is a forward or adjoint operation.
        """
        if self.compute:
            # Forward morphism - bra transforms ket
            result = (self.bra << 4) | other.ket
        else:
            # Adjoint morphism - ket transforms bra
            result = (other.bra << 4) | self.ket
            
        return BitMorphology(result)
    
    def inner_product(self, other: 'BitMorphology') -> int:
        """
        Compute the inner product as a measure of morphic compatibility.
        Higher values indicate greater resonance between morphologies.
        """
        # XOR the bras (difference = 0 means compatibility)
        bra_diff = self.bra ^ other.bra
        # XOR the kets (difference = 0 means compatibility)
        ket_diff = self.ket ^ other.ket
        
        # Count the number of matching bits (8 - number of differing bits)
        compatibility = 8 - bin(bra_diff).count('1') - bin(ket_diff).count('1')
        return compatibility
    
    def evolve(self) -> 'BitMorphology':
        """
        Autonomous evolution rule for this cellular automaton unit.
        Follows a morphic field principle based on internal structure.
        """
        # Example evolution rule: rotate values by compute bit
        if self.compute:
            # Rotate right - more "entropic" behavior
            new_raw = ((self.raw & 0x1) << 7) | (self.raw >> 1)
        else:
            # Rotate left - more "negentropic" behavior
            new_raw = ((self.raw << 1) & 0xFF) | ((self.raw >> 7) & 0x1)
            
        return BitMorphology(new_raw)
    
    @staticmethod
    def create_from_components(
        compute: bool,
        values: Tuple[bool, bool, bool],
        types: Tuple[bool, bool, bool, bool]
    ) -> 'BitMorphology':
        """Create a BitMorphology from component parts"""
        raw = (int(compute) << 7)
        raw |= (int(values[0]) << 6)
        raw |= (int(values[1]) << 5)
        raw |= (int(values[2]) << 4)
        raw |= (int(types[0]) << 3)
        raw |= (int(types[1]) << 2)
        raw |= (int(types[2]) << 1)
        raw |= int(types[3])
        return BitMorphology(raw)
    
    def __repr__(self) -> str:
        """Human-readable representation of the morphology"""
        c = "C" if self.compute else "c"
        v = "".join(["V" if v else "v" for v in self.value_triplet])
        t = "".join(["T" if t else "t" for t in self.type_quartet])
        return f"{c}{v}|{t} [0x{self.raw:02x}]"


class MorphicField:
    """
    A lattice of BitMorphology entities that evolve according to rulial dynamics.
    Functions as a distributed quantum-like computational fabric.
    """
    def __init__(self, size: Tuple[int, int]):
        self.width, self.height = size
        self.lattice = np.zeros((self.height, self.width), dtype=np.uint8)
        self.morphologies = [[BitMorphology(0) for _ in range(self.width)] 
                             for _ in range(self.height)]
        
    def initialize_random(self, seed: Optional[int] = None) -> None:
        """Initialize the field with random values"""
        rng = np.random.RandomState(seed)
        self.lattice = rng.randint(0, 256, (self.height, self.width), dtype=np.uint8)
        
        # Update the morphologies to match the raw values
        for y in range(self.height):
            for x in range(self.width):
                self.morphologies[y][x] = BitMorphology(self.lattice[y, x])
    
    def initialize_kernel(self, kernel_type: str) -> None:
        """Initialize with a specific kernel pattern"""
        if kernel_type == "gaussian_white":
            # Gaussian white noise centered around 128
            self.lattice = np.random.normal(128, 40, (self.height, self.width))
            self.lattice = np.clip(self.lattice, 0, 255).astype(np.uint8)
        elif kernel_type == "quine":
            # Self-replicating pattern - simple example
            # Creates a pattern that tends to recreate itself under the evolution rules
            midpoint = (self.width // 2, self.height // 2)
            for y in range(self.height):
                for x in range(self.width):
                    # Distance from center
                    dist = ((x - midpoint[0])**2 + (y - midpoint[1])**2)**0.5
                    # Create a radial pattern with alternating compute bits
                    compute = (int(dist) % 2 == 0)
                    values = (dist % 3 < 1, dist % 5 < 2, dist % 7 < 3)
                    types = (dist % 2 < 1, dist % 3 < 1, dist % 5 < 2, dist % 7 < 3)
                    
                    morph = BitMorphology.create_from_components(compute, values, types)
                    self.lattice[y, x] = morph.raw
                    self.morphologies[y][x] = morph
        
        # Update the morphologies to match the raw values
        for y in range(self.height):
            for x in range(self.width):
                self.morphologies[y][x] = BitMorphology(self.lattice[y, x])
    
    def step(self) -> None:
        """Evolve the entire field one step forward"""
        new_lattice = np.zeros_like(self.lattice)
        
        for y in range(self.height):
            for x in range(self.width):
                # Get the current morphology
                current = self.morphologies[y][x]
                
                # Get neighbors
                neighbors = self._get_neighbors(x, y)
                
                # Apply rulial dynamics
                new_morphology = self._apply_rulial_dynamics(current, neighbors)
                
                # Update the new lattice
                new_lattice[y, x] = new_morphology.raw
        
        # Update the lattice and morphologies
        self.lattice = new_lattice
        for y in range(self.height):
            for x in range(self.width):
                self.morphologies[y][x] = BitMorphology(self.lattice[y, x])
    
    def _get_neighbors(self, x: int, y: int) -> List[BitMorphology]:
        """Get the neighboring morphologies"""
        neighbors = []
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip the center
                
                nx, ny = (x + dx) % self.width, (y + dy) % self.height
                neighbors.append(self.morphologies[ny][nx])
        
        return neighbors
    
    def _apply_rulial_dynamics(self, center: BitMorphology, 
                              neighbors: List[BitMorphology]) -> BitMorphology:
        """
        Apply rulial dynamics based on the center and its neighbors.
        This is where the quantum-like probabilistic behavior emerges.
        """
        # First, find the neighbor with highest morphic resonance (inner product)
        max_resonance = -1
        most_resonant = None
        
        for neighbor in neighbors:
            resonance = center.inner_product(neighbor)
            if resonance > max_resonance:
                max_resonance = resonance
                most_resonant = neighbor
        
        # Apply different dynamics based on resonance levels
        if max_resonance >= 6:
            # High resonance - perform morphic transformation
            return center.morph(most_resonant)
        elif max_resonance >= 4:
            # Medium resonance - autonomous evolution
            return center.evolve()
        else:
            # Low resonance - maintain identity
            return BitMorphology(center.raw)
    
    def visualize(self) -> np.ndarray:
        """
        Create a visualization of the morphic field.
        Returns an RGB image representing the field state.
        """
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for y in range(self.height):
            for x in range(self.width):
                morph = self.morphologies[y][x]
                
                # Use different color channels for different aspects
                # Red channel: Compute bit
                image[y, x, 0] = 255 if morph.compute else 0
                
                # Green channel: Value bits
                v_sum = sum(int(v) for v in morph.value_triplet)
                image[y, x, 1] = int(v_sum * 255 / 3)
                
                # Blue channel: Type bits
                t_sum = sum(int(t) for t in morph.type_quartet)
                image[y, x, 2] = int(t_sum * 255 / 4)
        
        return image

def test_bit_morphology_system():
    # Create a small morphic field
    field = MorphicField((32, 32))
    
    # Initialize with a quine-like kernel
    field.initialize_kernel("quine")
    
    # Evolve the field for 100 steps
    initial_entropy = morphic_loss_function(field)
    print(f"Initial morphic entropy: {initial_entropy:.4f}")
    
    for i in range(100):
        field.step()
        
        if i % 10 == 0:
            entropy = morphic_loss_function(field)
            print(f"Step {i}, morphic entropy: {entropy:.4f}")
    
    final_entropy = morphic_loss_function(field)
    print(f"Final morphic entropy: {final_entropy:.4f}")
    
    # Visualize the field
    image = field.visualize()
    # Save or display the image

if __name__ == "__main__":
    test_bit_morphology_system()