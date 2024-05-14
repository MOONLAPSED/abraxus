# src/atom.py

from abc import ABC, abstractmethod
from typing import Union, List
import base64
import struct

class Atom(ABC):
    """Base class for atomic units."""
    @abstractmethod
    def __repr__(self):
        """Return a canonical representation of the atomic unit."""
        pass

class DataUnit(Atom):
    """A generic data unit capable of representing various data forms."""
    
    def __init__(self, data: Union[str, bytes, List[float], List[int]]):
        if isinstance(data, str):
            self.data_type = 'string'
            self.data = data.encode()  # Store as bytes internally
        elif isinstance(data, bytes):
            self.data_type = 'bytes'
            self.data = data
        elif isinstance(data, list) and all(isinstance(item, float) for item in data):
            self.data_type = 'embedding'
            self.data = data
        elif isinstance(data, list) and all(isinstance(item, int) for item in data):
            self.data_type = 'token'
            self.data = data
        else:
            raise ValueError("Unsupported data type.")

    def __repr__(self):
        if self.data_type == 'string':
            return f"DataUnit(string: {self.data.decode()})"
        elif self.data_type == 'bytes':
            return f"DataUnit(bytes: {self.data.hex()})"
        elif self.data_type == 'embedding':
            return f"DataUnit(embedding: {self.data})"
        elif self.data_type == 'token':
            return f"DataUnit(token: {self.data})"
        else:
            return "DataUnit(unsupported type)"

    def to_string(self) -> str:
        """Convert the data to a string."""
        if self.data_type in ['string', 'bytes']:
            return self.data.decode()
        elif self.data_type == 'embedding':
            return ', '.join(map(str, self.data))
        elif self.data_type == 'token':
            return ' '.join(map(str, self.data))
        else:
            raise ValueError("Unsupported data type.")

    def to_bytes(self) -> bytes:
        """Convert the data to bytes."""
        if self.data_type in ['string', 'bytes']:
            return self.data
        elif self.data_type == 'embedding':
            return struct.pack('f'*len(self.data), *self.data)
        elif self.data_type == 'token':
            return bytes(self.data)
        else:
            raise ValueError("Unsupported data type.")

    def to_base64(self) -> str:
        """Convert the data to a base64-encoded string."""
        return base64.b64encode(self.to_bytes()).decode()

    def to_hex(self) -> str:
        """Convert the data to a hexadecimal string."""
        return self.to_bytes().hex()

    def __and__(self, other: "DataUnit") -> "DataUnit":
        """Bitwise AND operation for byte data."""
        if self.data_type == 'bytes' and other.data_type == 'bytes':
            value = bytes(a & b for a, b in zip(self.data, other.data))
            return DataUnit(value)
        else:
            raise TypeError("AND operation only supported for byte data.")

    def __or__(self, other: "DataUnit") -> "DataUnit":
        """Bitwise OR operation for byte data."""
        if self.data_type == 'bytes' and other.data_type == 'bytes':
            value = bytes(a | b for a, b in zip(self.data, other.data))
            return DataUnit(value)
        else:
            raise TypeError("OR operation only supported for byte data.")