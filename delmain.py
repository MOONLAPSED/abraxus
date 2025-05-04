"""
Morphological Computing System
------------------------------
A framework for quantum-inspired computational morphisms with holographic properties.
This system unifies several computational concepts:
- Quantum-like state representations (superposition, entanglement)
- Type-theoretic morphisms with co/contravariance
- Byte-level semantics for computational state
- Transducer-based functional programming patterns
"""

import enum
import math
import time
import random
import sys
import functools
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, 
    TypeVar, Union, Generic, cast
)

# ============================================================================
# Type Variables and Base Types
# ============================================================================

# Type variables for generic programming
T = TypeVar('T')  # Type structure
V = TypeVar('V')  # Value space
R = TypeVar('R')  # Result type

# Covariant/contravariant type variables for advanced type modeling
T_co = TypeVar('T_co', covariant=True)  # Covariant Type structure
V_co = TypeVar('V_co', covariant=True)  # Covariant Value space
C_co = TypeVar('C_co', bound=Callable[..., Any], covariant=True)  # Covariant Control space

T_anti = TypeVar('T_anti', contravariant=True)  # Contravariant Type structure
V_anti = TypeVar('V_anti', contravariant=True)  # Contravariant Value space
C_anti = TypeVar('C_anti', bound=Callable[..., Any], contravariant=True)  # Contravariant Computation space

# BYTE type for byte-level operations
BYTE = TypeVar("BYTE", bound="BYTE_WORD")

# ============================================================================
# Core Enums and Constants
# ============================================================================

class Morphology(enum.Enum):
    """
    Represents the floor morphic state of a BYTE_WORD.
    
    C = 0: Floor morphic state (stable, low-energy)
    C = 1: Dynamic or high-energy state
    
    The control bit (C) indicates whether other holoicons can point to this holoicon:
    - DYNAMIC (1): Other holoicons CAN point to this holoicon
    - MORPHIC (0): Other holoicons CANNOT point to this holoicon
    
    This ontology maps to thermodynamic character: intensive & extensive.
    A 'quine' (self-instantiated runtime) is a low-energy, intensive system,
    while a dynamic holoicon is a high-energy, extensive system inherently
    tied to its environment.
    """
    MORPHIC = 0      # Stable, low-energy state
    DYNAMIC = 1      # High-energy, potentially transformative state
    
    # Fundamental computational orientation and symmetry
    MARKOVIAN = -1    # Forward-evolving, irreversible
    NON_MARKOVIAN = math.e  # Reversible, with memory


class QuantumState(enum.Enum):
    """Represents a computational state that tracks its quantum-like properties."""
    SUPERPOSITION = 1   # Known by handle only
    ENTANGLED = 2       # Referenced but not loaded
    COLLAPSED = 4       # Fully materialized
    DECOHERENT = 8      # Garbage collected


class WordSize(enum.IntEnum):
    """Standardized computational word sizes"""
    BYTE = 1     # 8-bit
    SHORT = 2    # 16-bit
    INT = 4      # 32-bit
    LONG = 8     # 64-bit


# ============================================================================
# Complex Number with Morphic Properties
# ============================================================================

class MorphicComplex:
    """Represents a complex number with morphic properties."""
    def __init__(self, real: float, imag: float):
        self.real = real
        self.imag = imag
    
    def conjugate(self) -> 'MorphicComplex':
        """Return the complex conjugate."""
        return MorphicComplex(self.real, -self.imag)
    
    def __add__(self, other: 'MorphicComplex') -> 'MorphicComplex':
        return MorphicComplex(self.real + other.real, self.imag + other.imag)
    
    def __mul__(self, other: 'MorphicComplex') -> 'MorphicComplex':
        return MorphicComplex(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real
        )
    
    def __repr__(self) -> str:
        if self.imag == 0:
            return f"{self.real}"
        elif self.real == 0:
            return f"{self.imag}j"
        else:
            sign = "+" if self.imag >= 0 else ""
            return f"{self.real}{sign}{self.imag}j"


# ============================================================================
# Base Byte-level Representation
# ============================================================================

class BYTE_WORD:
    """Basic 8-bit word representation."""
    def __init__(self, value: int = 0):
        if not 0 <= value <= 255:
            raise ValueError("BYTE_WORD value must be between 0 and 255")
        self.value = value

    def __repr__(self) -> str:
        return f"BYTE_WORD(value={self.value:08b})"


class ByteWord:
    """
    Enhanced representation of an 8-bit BYTE_WORD with a comprehensive interpretation of its structure.
    
    Bit Decomposition:
    - T (4 bits): State or data field
    - V (3 bits): Morphism selector or transformation rule
    - C (1 bit): Floor morphic state (pointability)
    """
    def __init__(self, raw: int):
        """
        Initialize a ByteWord from its raw 8-bit representation.
        
        Args:
            raw (int): 8-bit integer representing the BYTE_WORD
        """
        if not 0 <= raw <= 255:
            raise ValueError("ByteWord must be an 8-bit integer (0-255)")
            
        self.raw = raw
        self.value = raw & 0xFF  # Ensure 8-bit resolution
        
        # Decompose the raw value
        self.state_data = (raw >> 4) & 0x0F       # High nibble (4 bits)
        self.morphism = (raw >> 1) & 0x07         # Middle 3 bits
        self.floor_morphic = Morphology(raw & 0x01)  # Least significant bit
        
        self._refcount = 1
        self._state = QuantumState.SUPERPOSITION

    @property
    def pointable(self) -> bool:
        """
        Determine if other holoicons can point to this holoicon.
        
        Returns:
            bool: True if the holoicon is in a dynamic (pointable) state
        """
        return self.floor_morphic == Morphology.DYNAMIC

    def __repr__(self) -> str:
        return f"ByteWord(T={self.state_data:04b}, V={self.morphism:03b}, C={self.floor_morphic.value})"

    @staticmethod
    def xnor(a: int, b: int, width: int = 4) -> int:
        """Perform a bitwise XNOR operation."""
        return ~(a ^ b) & ((1 << width) - 1)  # Mask to width-bit output

    @staticmethod
    def abelian_transform(t: int, v: int, c: int) -> int:
        """
        Perform the XNOR-based Abelian transformation.
        
        Args:
            t: State/data field
            v: Morphism selector
            c: Floor morphic state
            
        Returns:
            Transformed state value
        """
        if c == 1:
            return ByteWord.xnor(t, v)  # Apply XNOR transformation
        return t  # Identity morphism when c = 0

    @staticmethod
    def extract_lsb(state: Union[str, int, bytes], word_size: int) -> Any:
        """
        Extract least significant bit/byte based on word size.
        
        Args:
            state: Input value to extract from
            word_size: Size of word to determine extraction method
            
        Returns:
            Extracted least significant bit/byte
        """
        if word_size == 1:
            return state[-1] if isinstance(state, str) else str(state)[-1]
        elif word_size == 2:
            return (
                state & 0xFF if isinstance(state, int) else
                state[-1] if isinstance(state, bytes) else
                state.encode()[-1]
            )
        elif word_size >= 3:
            return hashlib.sha256(
                state.encode() if isinstance(state, str) else 
                state if isinstance(state, bytes) else
                str(state).encode()
            ).digest()[-1]


# ============================================================================
# Quantum-like State Implementation
# ============================================================================

class QuantumStateImpl:
    """Implementation of quantum-like state operations."""
    
    def __init__(self, amplitudes: List[MorphicComplex], dimension: int):
        """
        Initialize a quantum state with complex amplitudes.
        
        Args:
            amplitudes: List of complex amplitudes
            dimension: Dimension of the Hilbert space
        """
        self.amplitudes = amplitudes
        self.dimension = dimension
        self._normalize()
    
    def _normalize(self) -> None:
        """Normalize the quantum state to ensure probabilities sum to 1."""
        norm_squared = sum(amp.real**2 + amp.imag**2 for amp in self.amplitudes)
        norm = math.sqrt(norm_squared)
        
        if norm > 0:
            for i in range(len(self.amplitudes)):
                self.amplitudes[i] = MorphicComplex(
                    self.amplitudes[i].real / norm,
                    self.amplitudes[i].imag / norm
                )
    
    def measure(self) -> int:
        """
        Perform a measurement on the quantum state.
        Returns the index of the basis state that was measured.
        """
        # Calculate probabilities for each basis state
        probabilities = []
        for amp in self.amplitudes:
            # Probability is |amplitude|²
            prob = amp.real**2 + amp.imag**2
            probabilities.append(prob)
            
        # Simulate measurement using the probabilities
        r = random.random()
        cumulative_prob = 0
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if r <= cumulative_prob:
                return i
                
        # Fallback (shouldn't happen with normalized state)
        return len(self.amplitudes) - 1
    
    def superposition(self, other: 'QuantumStateImpl', coeff1: MorphicComplex, coeff2: MorphicComplex) -> 'QuantumStateImpl':
        """
        Create a superposition of two quantum states.
        |ψ⟩ = a|ψ₁⟩ + b|ψ₂⟩
        """
        if self.dimension != other.dimension:
            raise ValueError("Quantum states must belong to same Hilbert space")
            
        new_amplitudes = []
        for i in range(len(self.amplitudes)):
            new_amp = (self.amplitudes[i] * coeff1) + (other.amplitudes[i] * coeff2)
            new_amplitudes.append(new_amp)
            
        return QuantumStateImpl(new_amplitudes, self.dimension)
    
    def entangle(self, other: 'QuantumStateImpl') -> 'QuantumStateImpl':
        """
        Create an entangled state from two quantum states.
        |ψ⟩ = (|ψ₁⟩|0⟩ + |ψ₂⟩|1⟩)/√2
        This is a simplified version of entanglement for demonstration.
        """
        # For simplicity, we'll just return a superposition
        coeff = MorphicComplex(1/math.sqrt(2), 0)
        return self.superposition(other, coeff, coeff)


# ============================================================================
# Python Object Abstraction
# ============================================================================

class PyObjABC(ABC):
    """Abstract Base Class for PyObject-like objects (including __Atom__)."""
    
    @abstractmethod
    def __getattribute__(self, name: str) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def __setattr__(self, name: str, value: Any) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def __class__(self) -> type:
        raise NotImplementedError
    
    @property
    def ob_refcnt(self) -> int:
        """Returns the object's reference count."""
        return self._refcount
    
    @ob_refcnt.setter
    def ob_refcnt(self, value: int) -> None:
        """Sets the object's reference count."""
        self._refcount = value
    
    @property
    def ob_ttl(self) -> Optional[int]:
        """Returns the object's time-to-live (in seconds or None)."""
        return self._ttl
    
    @ob_ttl.setter
    def ob_ttl(self, value: Optional[int]) -> None:
        """Sets the object's time-to-live."""
        self._ttl = value


@dataclass
class CPythonFrame:
    """
    Quantum-informed object representation.
    Maps conceptually to CPython's PyObject structure.
    """
    type_ptr: int             # Memory address of type object
    obj_type: Type[Any]           # Type information
    _value: Any                # Actual value (renamed to avoid conflict with property)
    refcount: int = field(default=1)
    ttl: Optional[int] = None
    _state: QuantumState = field(init=False, repr=False)
    
    def __post_init__(self):
        """Initialize with timestamp and quantum properties"""
        self._birth_timestamp = time.time()
        self._refcount = self.refcount
        self._ttl = self.ttl 
        
        # Initialize _state to default SUPERPOSITION
        self._state = QuantumState.SUPERPOSITION
        
        if self.ttl is not None:
            self._ttl_expiration = self._birth_timestamp + self.ttl
        else:
            self._ttl_expiration = None

        # self._value is already set by dataclass init

        if self._state == QuantumState.SUPERPOSITION:
            self._superposition = [self._value]
            self._superposition_timestamp = time.time()
        else:
            self._superposition = None
            
        if self._state == QuantumState.ENTANGLED:
            self._entanglement = [self._value]
            self._entanglement_timestamp = time.time()
        else:
            self._entanglement = None
            
        if self.obj_type.__module__ == 'builtins':
            """All 'knowledge' aka data is treated as python modules and these are the flags for controlling what is canon."""
            self._is_primitive = True
            self._primitive_type = self.obj_type.__name__
            self._primitive_value = self._value
        else:
            self._is_primitive = False
    
    @classmethod
    def from_object(cls, obj: object) -> 'CPythonFrame':
        """Extract CPython frame data from any Python object"""
        return cls(
            type_ptr=id(type(obj)),
            obj_type=type(obj), # Use obj_type
            _value=obj,
            refcount=sys.getrefcount(obj) - 1 # Initial refcount estimate
            # Let ttl and state use their defaults unless specified
        )
    
    @property
    def value(self) -> Any:
        """Get the current value, potentially collapsing state."""
        if self._state == QuantumState.SUPERPOSITION:
            return random.choice(self._superposition)
        return self._value
    
    @property
    def state(self) -> QuantumState:
        """Current quantum-like state"""
        return self._state
    
    @state.setter
    def state(self, new_state: QuantumState) -> None:
        """Set the quantum state with appropriate side effects"""
        old_state = self._state
        self._state = new_state
        
        # Handle state transition side effects
        if new_state == QuantumState.COLLAPSED and old_state == QuantumState.SUPERPOSITION:
            if self._superposition:
                self._value = random.choice(self._superposition)
                self._superposition = None
    
    def collapse(self) -> Any:
        """Force state resolution"""
        if self._state != QuantumState.COLLAPSED:
            if self._state == QuantumState.SUPERPOSITION and self._superposition:
                self._value = random.choice(self._superposition) # Update internal value
                self._superposition = None # Clear superposition list
            elif self._state == QuantumState.ENTANGLED:
                 # Decide how entanglement collapses - maybe pick from list?
                 if self._entanglement:
                     self._value = random.choice(self._entanglement) # Example resolution
                 self._entanglement = None # Clear entanglement list
            self._state = QuantumState.COLLAPSED
        return self._value # Return the now-collapsed internal value

    
    def entangle_with(self, other: 'CPythonFrame') -> None:
        """Create quantum entanglement with another object.""" 
        if self._entanglement is None:
            self._entanglement = [self.value]
        if other._entanglement is None:
            other._entanglement = [other.value]
            
        self._entanglement.extend(other._entanglement)
        other._entanglement = self._entanglement
        self._state = other._state = QuantumState.ENTANGLED
    
    def check_ttl(self) -> bool:
        """Check if TTL expired and collapse state if necessary."""
        if self.ttl is not None and self._ttl_expiration is not None and time.time() >= self._ttl_expiration:
            self.collapse()
            return True
        return False
    
    def observe(self) -> Any:
        """Collapse state upon observation if necessary."""
        self.check_ttl()
        if self._state == QuantumState.SUPERPOSITION:
            self._state = QuantumState.COLLAPSED
            if self._superposition:
                self._value = random.choice(self._superposition)
        elif self._state == QuantumState.ENTANGLED:
            self._state = QuantumState.COLLAPSED
        return self.value


# ============================================================================
# Functional Programming Patterns - Transducers
# ============================================================================

class Missing:
    """Marker class to indicate a missing value."""
    pass


class Reduced:
    """Sentinel class to signal early termination during reduction."""
    def __init__(self, val: Any):
        self.val = val


def ensure_reduced(x: Any) -> Union[Any, Reduced]:
    """Ensure the value is wrapped in a Reduced sentinel."""
    return x if isinstance(x, Reduced) else Reduced(x)


def unreduced(x: Any) -> Any:
    """Unwrap a Reduced value or return the value itself."""
    return x.val if isinstance(x, Reduced) else x


def reduce(function: Callable[[Any, T], Any], iterable: Iterable[T], initializer: Any = Missing) -> Any:
    """A custom reduce implementation that supports early termination with Reduced."""
    accum_value = initializer if initializer is not Missing else function()
    for x in iterable:
        accum_value = function(accum_value, x)
        if isinstance(accum_value, Reduced):
            return accum_value.val
    return accum_value


class Transducer:
    """Base class for defining transducers."""
    def __init__(self, step: Callable[[Any, T], Any]):
        self.step = step

    def __call__(self, step: Callable[[Any, T], Any]) -> Callable[[Any, T], Any]:
        """The transducer's __call__ method allows it to be used as a decorator."""
        return self.step(step)


class Map(Transducer):
    """Transducer for mapping elements with a function."""
    def __init__(self, f: Callable[[T], R]):
        def _map_step(step):
            def new_step(r: Any = Missing, x: Optional[T] = Missing):
                if r is Missing:
                    return step()
                if x is Missing:
                    return step(r)
                return step(r, f(x))
            return new_step
        super().__init__(_map_step)


class Filter(Transducer):
    """Transducer for filtering elements based on a predicate."""
    def __init__(self, pred: Callable[[T], bool]):
        def _filter_step(step):
            def new_step(r: Any = Missing, x: Optional[T] = Missing):
                if r is Missing:
                    return step()
                if x is Missing:
                    return step(r)
                return step(r, x) if pred(x) else r
            return new_step
        super().__init__(_filter_step)


class Cat(Transducer):
    """Transducer for flattening nested collections."""
    def __init__(self):
        def _cat_step(step):
            def new_step(r: Any = Missing, x: Optional[Any] = Missing):
                if r is Missing:
                    return step()
                if x is Missing:
                    return step(r)
                    
                if not hasattr(x, '__iter__'):
                    raise TypeError(f"Expected iterable, got {type(x)}")
                    
                result = r
                for item in x:
                    result = step(result, item)
                    if isinstance(result, Reduced):
                        return result
                return result
            return new_step
        super().__init__(_cat_step)


def compose(*fns: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Compose functions in reverse order."""
    return functools.reduce(lambda f, g: lambda x: f(g(x)), fns)


def transduce(xform: Transducer, f: Callable[[Any, T], Any], start: Any, coll: Iterable[T]) -> Any:
    """Apply a transducer to a collection with an initial value."""
    reducer = xform(f)
    return reduce(reducer, coll, start)


def mapcat(f: Callable[[T], Iterable[R]]) -> Transducer:
    """Map then flatten results into one collection."""
    return compose(Map(f), Cat())


def into(target: Union[list, set], xducer: Transducer, coll: Iterable[T]) -> Any:
    """Apply transducer and collect results into a target container."""
    def append(r: Any = Missing, x: Optional[Any] = Missing) -> Any:
        """Append to a collection."""
        if r is Missing:
            return []
        if hasattr(r, 'append'):
            r.append(x)
        elif hasattr(r, 'add'):
            r.add(x)
        return r
        
    return transduce(xducer, append, target, coll)


# ============================================================================
# Morphological Framework - Rule-based Transformations
# ============================================================================

class MorphologicalRule:
    """Rule that maps structural transformations in code morphologies."""
    
    def __init__(self, symmetry: str, conservation: str, lhs: str, rhs: List[Union[str, Morphology, ByteWord]]):
        self.symmetry = symmetry
        self.conservation = conservation
        self.lhs = lhs
        self.rhs = rhs
    
    def apply(self, input_seq: List[str]) -> List[str]:
        """Applies the morphological transformation to an input sequence."""
        if self.lhs in input_seq:
            idx = input_seq.index(self.lhs)
            return input_seq[:idx] + [str(elem) for elem in self.rhs] + input_seq[idx + 1:]
        return input_seq


@dataclass
class MorphologicPyOb:
    """
    The unification of Morphologic transformations and PyOb behavior.
    This is the grandparent class for all runtime polymorphs.
    It encapsulates stateful, structural, and computational potential.
    """
    symmetry: str
    conservation: str
    lhs: str
    rhs: List[Union[str, 'Morphology']]
    value: Any
    type_ptr: int = field(default_factory=lambda: id(object))
    ttl: Optional[int] = None
    state: QuantumState = field(default=QuantumState.SUPERPOSITION)
    
    def __post_init__(self):
        self._refcount = 1
        self._birth_timestamp = time.time()
        self._state = self.state
        
        if self.ttl is not None:
            self._ttl_expiration = self._birth_timestamp + self.ttl
        else:
            self._ttl_expiration = None
            
        if self.state == QuantumState.SUPERPOSITION:
            self._superposition = [self.value]
        else:
            self._superposition = None
            
        if self.state == QuantumState.ENTANGLED:
            self._entanglement = [self.value]
        else:
            self._entanglement = None
    
    def apply_transformation(self, input_seq: List[str]) -> List[str]:
        """
        Applies morphological transformation while preserving object state.
        """
        if self.lhs in input_seq:
            idx = input_seq.index(self.lhs)
            transformed = input_seq[:idx] + [str(elem) for elem in self.rhs] + input_seq[idx + 1:]
            self._state = QuantumState.ENTANGLED
            return transformed
        return input_seq
    
    def collapse(self) -> Any:
        """Collapse to resolved state."""
        if self._state != QuantumState.COLLAPSED:
            if self._state == QuantumState.SUPERPOSITION and self._superposition:
                self.value = random.choice(self._superposition)
            self._state = QuantumState.COLLAPSED
        return self.value
    
    def collapse_and_transform(self) -> Any:
        """Collapse to resolved state and apply morphological transformation to value."""
        collapsed_value = self.collapse()
        if isinstance(collapsed_value, list):
            return self.apply_transformation(collapsed_value)
        return collapsed_value
    
    def entangle_with(self, other: 'MorphologicPyOb') -> None:
        """Entangle with another MorphologicPyOb to preserve state & entanglement symmetry in Morphologic terms."""
        if self._entanglement is None:
            self._entanglement = [self.value]
        if other._entanglement is None:
            other._entanglement = [other.value]
            
        self._entanglement.extend(other._entanglement)
        other._entanglement = self._entanglement
        
        if self.lhs == other.lhs and self.conservation == other.conservation:
            self._state = QuantumState.ENTANGLED
            other._state = QuantumState.ENTANGLED


# ============================================================================
# Examples and Demo
# ============================================================================

def demo():
    """Demonstrate the functionality of the morphological computing system."""
    print("Morphological Computing System Demo")
    print("===================================")
    
    # Create ByteWord example
    print("\n1. ByteWord Representation")
    byte_val = 0b10110101  # Binary: 10110101
    byte_word = ByteWord(byte_val)
    print(f"ByteWord: {byte_word}")
    print(f"Is pointable: {byte_word.pointable}")
    
    # Abelian transform example
    t_val = byte_word.state_data
    v_val = byte_word.morphism
    c_val = byte_word.floor_morphic.value
    transformed = ByteWord.abelian_transform(t_val, v_val, c_val)
    print(f"Abelian transform of {t_val:04b} with V={v_val:03b}, C={c_val}: {transformed:04b}")
    
    # Quantum state example
    print("\n2. Quantum State Simulation")
    state1 = QuantumStateImpl([
        MorphicComplex(0.7071, 0),
        MorphicComplex(0, 0.7071)
    ], 2)
    
    state2 = QuantumStateImpl([
        MorphicComplex(0, 0.7071),
        MorphicComplex(0.7071, 0)
    ], 2)
    
    # Perform measurements
    print("Measuring state1 (10 times):")
    measurements = [state1.measure() for _ in range(10)]
    print(f"Results: {measurements}")
    
    # Create superposition
    superpos = state1.superposition(
        state2, 
        MorphicComplex(0.5, 0.5), 
        MorphicComplex(0.5, -0.5)
    )
    print("Superposition amplitudes:")
    for amp in superpos.amplitudes:
        print(f"  {amp}")
    
    # Morphological rule example
    print("\n3. Morphological Transformation")
    rule = MorphologicalRule(
        symmetry="reflection",
        conservation="information",
        lhs="lambda",
        rhs=["def", "anonymous", ":"]
    )
    
    code_seq = ["import", "sys", "lambda", "x", ":", "x+1"]
    transformed_seq = rule.apply(code_seq)
    print(f"Original: {code_seq}")
    print(f"Transformed: {transformed_seq}")
    
    # CPythonFrame example
    print("\n4. CPythonFrame Quantum Object")
    frame = CPythonFrame(
        type_ptr=id(str),
        obj_type=str,
        _value="quantum text"
    )
    frame.state = QuantumState.SUPERPOSITION
    
    print(f"Initial state: {frame.state}")
    print(f"Observed value: {frame.observe()}")
    print(f"State after observation: {frame.state}")
    
    # Transducer example
    print("\n5. Transducer Composition")
    numbers = list(range(10))
    
    # Create a composite transducer: filter even numbers and double them
    xform = compose(
        Filter(lambda x: x % 2 == 0),
        Map(lambda x: x * 2)
    )
    
    result = into([], xform, numbers)
    print(f"Original: {numbers}")
    print(f"Filtered and doubled: {result}")

if __name__ == "__main__":
    demo()
