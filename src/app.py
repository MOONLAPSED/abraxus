# /src/app.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, TypeVar
import functools
import struct

T = TypeVar('T')

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

@dataclass(frozen=True)
class AtomDataclass(Generic[T], Atom):
    value: T
    data: T
    data_type: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'data_type', self._determine_data_type(self.data))

    @staticmethod
    def _determine_data_type(data: Any) -> str:
        data_type = type(data).__name__
        if data_type == 'str':
            return 'string'
        elif data_type == 'int':
            return 'integer'
        elif data_type == 'float':
            return 'float'
        elif data_type == 'bool':
            return 'boolean'
        elif data_type == 'list':
            return 'list'
        elif data_type == 'dict':
            return 'dictionary'
        else:
            return 'unknown'

    def to_dataclass(self):
        return self

    def encode(self) -> bytes:
        data_type_bytes = self.data_type.encode('utf-8')
        data_bytes = self._encode_data()
        header = struct.pack('!I', len(data_bytes))
        return header + data_type_bytes + data_bytes
    
    def _encode_data(self) -> bytes:
        if self.data_type == 'string':
            return self.data.encode('utf-8')
        elif self.data_type == 'integer':
            return struct.pack('!q', self.data)
        elif self.data_type == 'float':
            return struct.pack('!d', self.data)
        elif self.data_type == 'boolean':
            return struct.pack('?', self.data)
        elif self.data_type == 'list':
            return b''.join([AtomDataclass(element)._encode_data() for element in self.data])
        elif self.data_type == 'dictionary':
            return b''.join(
                [AtomDataclass(key)._encode_data() + AtomDataclass(value)._encode_data() for key, value in self.data.items()]
            )
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
    
    def execute(self, *args, **kwargs) -> Any:
        pass

    def decode(self, data: bytes) -> None:
        pass

    def __repr__(self):
        return f"AtomDataclass(value={self.value}, data={self.data}, data_type={self.data_type})"

@dataclass
class FormalTheory(Generic[T]):
    reflexivity: Callable[[T], bool] = lambda x: x == x
    symmetry: Callable[[T, T], bool] = lambda x, y: x == y
    transitivity: Callable[[T, T, T], bool] = lambda x, y, z: (x == y) and (y == z) and (x == z)
    transparency: Callable[[Callable[..., T], T, T], T] = lambda f, x, y: f(x) if x == y else None
    case_base: Dict[str, Callable[[T, T], T]] = field(default_factory=dict)
    
    def __post_init__(self):
        self.case_base = {
            '⊤': lambda x, _: x,
            '⊥': lambda _, y: y,
            'a': functools.partial(self.if_else, a=True)
        }
    
    @staticmethod
    def if_else(a: bool, x: T, y: T) -> T:
        return x if a else y

    def compare(self, atoms: list) -> bool:
        comparison = []
        for i in range(1, len(atoms)):
            comparison.append(self.symmetry(atoms[0].value, atoms[i].value))
        return all(comparison)

    def __repr__(self):
        return (f"FormalTheory(\n"
                f"  reflexivity={self.reflexivity},\n"
                f"  symmetry={self.symmetry},\n"
                f"  transitivity={self.transitivity},\n"
                f"  transparency={self.transparency},\n"
                f"  case_base={self.case_base}\n"
                f")")