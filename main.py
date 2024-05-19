from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, TypeVar, Any, List
import struct

# Type Variables
T = TypeVar('T')

# Abstract Base Class
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

# Dataclass Implementation
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

    def encode(self) -> bytes:
        data_type_bytes = self.data_type.encode('utf-8')
        data_bytes = self._encode_data()
        header = struct.pack('!I', len(data_type_bytes))
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

    def __repr__(self):
        return f"AtomDataclass(id={id(self)}, value={self.value}, data_type='{self.data_type}')"

    def to_dataclass(self):
        return self
    
    # Overloaded operators
    def __add__(self, other):
        return AtomDataclass(self.value + other.value)

    def __sub__(self, other):
        return AtomDataclass(self.value - other.value)

    def __mul__(self, other):
        return AtomDataclass(self.value * other.value)

    def __truediv__(self, other):
        return AtomDataclass(self.value / other.value)

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value

@dataclass
class FormalTheory(Atom, Generic[T]):
    reflexivity: Callable[[T], bool] = lambda x: x == x
    symmetry: Callable[[T, T], bool] = lambda x, y: x == y
    transitivity: Callable[[T, T, T], bool] = lambda x, y, z: (x == y) and (y == z) and (x == z)
    transparency: Callable[[Callable[..., T], T, T], T] = lambda f, x, y: f(True, x, y) if x == y else None
    case_base: Dict[str, Callable[..., bool]] = field(default_factory=dict)
    
    def __post_init__(self):
        self.case_base = {
            '⊤': lambda x, _: x,
            '⊥': lambda _, y: y,
            'a': self.if_else_a,
            '¬': lambda a: not a,
            '∧': lambda a, b: a and b,
            '∨': lambda a, b: a or b,
            '→': lambda a, b: (not a) or b,
            '↔': lambda a, b: (a and b) or (not a and not b),
            '¬∨': lambda a, b: not (a or b),  # NOR operation
            '¬∧': lambda a, b: not (a and b),  # NAND operation
            'contrapositive': self.contrapositive
        }
    
    def if_else(self, a: bool, x: T, y: T) -> T:
        return x if a else y

    def if_else_a(self, x: T, y: T) -> T:
        return self.if_else(True, x, y)

    def contrapositive(self, a: bool, b: bool) -> bool:
        return (not b) or (not a)
    
    def compare(self, atoms: List[AtomDataclass[T]]) -> bool:
        if not atoms:
            return False
        comparison = [self.symmetry(atoms[0].value, atoms[i].value) for i in range(1, len(atoms))]
        return all(comparison)
    
    def encode(self) -> bytes:
        return str(self.case_base).encode()

    def decode(self, data: bytes) -> None:
        # Example decoding for formal theory properties
        pass

    def execute(self, *args, **kwargs) -> Any:
        pass

    def to_dataclass(self):
        return super().to_dataclass()
    
    def __repr__(self):
        case_base_repr = {
            key: (value.__name__ if callable(value) else value)
            for key, value in self.case_base.items()
        }
        return (f"""FormalTheory(\n
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

    encoded = atom3._encode_data()
    print(f"Encoded atom3: {encoded}")

    decoder = encoded.decode()
    print(f"Decoded atom3: {type(decoder), decoder.__repr__(), decoder.__dir__()} size: {decoder.__sizeof__()} kb, memory address: {hex(id(decoder))}")

    theory = FormalTheory[int]()
    print(theory)
    print(f"Encoded theory: {theory.encode()}")

    atoms = [AtomDataclass(10), AtomDataclass(10), AtomDataclass(20)]
    comparison = theory.compare(atoms)
    print(f"Comparison result: {comparison}")

    contrapositive_result = theory.contrapositive(True, False)
    print(f"Contrapositive result: {contrapositive_result}")

if __name__ == "__main__":
    repl()