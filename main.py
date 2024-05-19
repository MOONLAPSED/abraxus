from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Generic, TypeVar, Any
from dataclasses import dataclass, field
import struct

# Generics
T = TypeVar('T')

# Abstracts
class Atom(ABC):
    @abstractmethod
    def encode(self) -> bytes:
        pass

    @abstractmethod
    def decode(self, data: bytes) -> None:
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def to_dataclass(self):
        pass

# Dataclasses
@dataclass(frozen=True)
class AtomDataclass(Generic[T], Atom):
    value: T
    data_type: str = field(init=False)

    def __post_init__(self):
        type_map = {
            'str': 'string',
            'int': 'integer',
            'float': 'float',
            'bool': 'boolean',
            'list': 'list',
            'dict': 'dictionary'
        }
        data_type_name = type(self.value).__name__
        object.__setattr__(self, 'data_type', type_map.get(data_type_name, 'unsupported'))

    def __repr__(self):
        return f"AtomDataclass(id={id(self)}, value={self.value}, data_type='{self.data_type}')"

    def to_dataclass(self):
        return self

    def __add__(self, other):
        return AtomDataclass(self.value + other.value)

    def __sub__(self, other):
        return AtomDataclass(self.value - other.value)

    def __mul__(self, other):
        return AtomDataclass(self.value * other.value)

    def __truediv__(self, other):
        return AtomDataclass(self.value / other.value)

    def __eq__(self, other):
        if isinstance(other, AtomDataclass):
            return self.value == other.value
        return False

    def __lt__(self, other):
        if isinstance(other, AtomDataclass):
            return self.value < other.value
        return False

    def __le__(self, other):
        if isinstance(other, AtomDataclass):
            return self.value <= other.value
        return False

    def __gt__(self, other):
        if isinstance(other, AtomDataclass):
            return self.value > other.value
        return False

    def __ge__(self, other):
        if isinstance(other, AtomDataclass):
            return self.value >= other.value
        return False

    def encode(self) -> bytes:
        data_type_bytes = self.data_type.encode('utf-8')
        data_bytes = self._encode_data()
        header = struct.pack('!I', len(data_bytes))
        return header + data_type_bytes + data_bytes
    
    def _encode_data(self) -> bytes:
        if self.data_type == 'string':
            return self.value.encode('utf-8')
        elif self.data_type == 'integer':
            return struct.pack('!q', self.value)
        elif self.data_type == 'float':
            return struct.pack('!d', self.value)
        elif self.data_type == 'boolean':
            return struct.pack('?', self.value)
        elif self.data_type == 'list':
            return b''.join([AtomDataclass(element)._encode_data() for element in self.value])
        elif self.data_type == 'dictionary':
            return b''.join(
                [AtomDataclass(key)._encode_data() + AtomDataclass(value)._encode_data() for key, value in self.value.items()]
            )
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
    
    def decode(self, data: bytes) -> None:
        header_length = struct.unpack('!I', data[:4])[0]
        data_type_bytes = data[4:4 + header_length]
        data_type = data_type_bytes.decode('utf-8')
        data_bytes = data[4 + header_length:]

        type_map_reverse = {
            'string': 'str',
            'integer': 'int',
            'float': 'float',
            'boolean': 'bool',
            'list': 'list',
            'dictionary': 'dict'
        }

        if data_type == 'string':
            value = data_bytes.decode('utf-8')
        elif data_type == 'integer':
            value = struct.unpack('!q', data_bytes)[0]
        elif data_type == 'float':
            value = struct.unpack('!d', data_bytes)[0]
        elif data_type == 'boolean':
            value = struct.unpack('?', data_bytes)[0]
        elif data_type == 'list':
            value = []
            offset = 0
            while offset < len(data_bytes):
                element = AtomDataclass(None)
                element_size = element.decode(data_bytes[offset:])
                value.append(element.value)
                offset += element_size
        elif data_type == 'dictionary':
            value = {}
            offset = 0
            while offset < len(data_bytes):
                key = AtomDataclass(None)
                key_size = key.decode(data_bytes[offset:])
                offset += key_size
                val = AtomDataclass(None)
                value_size = val.decode(data_bytes[offset:])
                offset += value_size
                value[key.value] = val.value
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        self.value = value
        object.__setattr__(self, 'data_type', type_map_reverse.get(data_type, 'unsupported'))

    def execute(self, *args, **kwargs) -> Any:
        pass

@dataclass
class FormalTheory(Atom, Generic[T]):
    """
    FormalTheory is a specialized class extending Atom to model theoretical constructs using reflexivity, symmetry, and transitivity as logical properties.
    Attributes:
        reflexivity, symmetry, transitivity, transparency: Callable lambdas for various logical operations.
        case_base: Default dictionary to hold case mappings.
    Methods:
        __post_init__(): Initializes the case_base with logic operations.
        if_else and if_else_a: Conditional logic operations.
        compare: Examines symmetry within a list of AtomDataclass.
        encode and decode: Handle encoding/decoding for formal theory properties.
        execute: Placeholder for operations execution.
    """
    reflexivity: Callable[[T], bool] = lambda x: x == x
    symmetry: Callable[[T, T], bool] = lambda x, y: x == y
    transitivity: Callable[[T, T, T], bool] = lambda x, y, z: (x == y) and (y == z) and (x == z)
    transparency: Callable[[Callable[..., T], T, T], T] = lambda f, x, y: f(True, x, y) if x == y else None
    case_base: Dict[str, Callable[[T, T], T]] = field(default_factory=dict)
    
    def __post_init__(self):
        """
        "This tension between theory and anti-theory can be seen as a driving force
        for scientific progress and the continuous refinement of theoretical frameworks."
        """
        self.case_base = {
            '⊤': lambda x, _: x,
            '⊥': lambda _, y: y,
            'a': self.if_else_a
        }

    def if_else(self, a: bool, x: T, y: T) -> T:
        return x if a else y

    def if_else_a(self, x: T, y: T) -> T:
        return self.if_else(True, x, y)

    def compare(self, atoms: List[AtomDataclass[T]]) -> bool:
        if not atoms:
            return False
        comparison = [self.symmetry(atoms[0].value, atoms[i].value) for i in range(1, len(atoms))]
        return all(comparison)

    def encode(self) -> bytes:
        # Example encoding for formal theory properties
        return str(self.case_base).encode()

    def decode(self, data: bytes) -> None:
        # Example decoding for formal theory properties
        pass

    def execute(self, *args, **kwargs) -> Any:
        # Placeholder
        pass

    def to_dataclass(self):
        return super().to_dataclass()

    def __repr__(self):
        case_base_repr = {
            key: (value.__name__ if callable(value) else value)
            for key, value in self.case_base.items()
        }
        return (f"""FormalTheory(\n
    "This tension between theory and anti-theory can be seen as a driving force
    for scientific progress and the continuous refinement of theoretical frameworks."\n
    reflexivity={self.reflexivity.__name__},\n
    symmetry={self.symmetry.__name__},\n
    transitivity={self.transitivity.__name__},\n
    transparency={self.transparency.__name__},\n
    case_base={case_base_repr}\n
    )""")


def repl():
    atom1 = AtomDataclass(10)
    atom2 = AtomDataclass(20)
    print(atom1)
    print(atom2)

    atom3 = atom1 + atom2
    print(f"atom1 + atom2 = {atom3}")

    encoded = atom3.encode()
    print(f"Encoded atom3: {encoded}")

    atom4 = AtomDataclass(0)
    print(f"Decoded atom4: {atom4}")

    theory = FormalTheory[int]()
    print(theory)

    atoms = [AtomDataclass(10), AtomDataclass(10)]
    comparison = theory.compare(atoms)
    print(f"Comparison result: {comparison}")

if __name__ == "__main__":
    repl()
