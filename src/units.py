from abc import ABC, abstractmethod
from typing import Union, List, Generic, TypeVar, Callable, Any
import base64
import struct
from dataclasses import dataclass, field
import operator
from functools import reduce, partial

# Data Management Classes
T = TypeVar('T')


class Atom(ABC):
    """Base class for atomic units."""

    @abstractmethod
    def __repr__(self):
        """Return a canonical representation of the atomic unit."""
        pass

    @abstractmethod
    def to_dataclass(self):
        """Convert the instance to a dataclass."""
        pass


class DataUnit(Atom):
    """A generic data unit capable of representing various data forms."""

    def __init__(self, data: Union[str, bytes, List[float], List[int]]):
        self.data_type = self._determine_data_type(data)
        self.data = self._encode_data(data)

    @staticmethod
    def _determine_data_type(data):
        if isinstance(data, str):
            return 'string'
        elif isinstance(data, bytes):
            return 'bytes'
        elif isinstance(data, list) and all(isinstance(item, float) for item in data):
            return 'embedding'
        elif isinstance(data, list) and all(isinstance(item, int) for item in data):
            return 'token'
        else:
            raise ValueError("Unsupported data type.")

    @staticmethod
    def _encode_data(data):
        if isinstance(data, str):
            return data.encode()
        return data

    def __repr__(self):
        if self.data_type == 'string':
            return f"DataUnit(string: {self.data.decode()})"
        elif self.data_type == 'bytes':
            return f"DataUnit(bytes: {self.data.hex()})"
        elif self.data_type == 'embedding':
            return f"DataUnit(embedding: {self.data})"
        elif self.data_type == 'token':
            return f"DataUnit(token: {self.data})"
        return "DataUnit(unsupported type)"

    def to_string(self) -> str:
        if self.data_type in ['string', 'bytes']:
            return self.data.decode()
        elif self.data_type == 'embedding':
            return ', '.join(map(str, self.data))
        elif self.data_type == 'token':
            return ' '.join(map(str, self.data))
        raise ValueError("Unsupported data type.")

    def to_bytes(self) -> bytes:
        if self.data_type in ['string', 'bytes']:
            return self.data
        elif self.data_type == 'embedding':
            return struct.pack('f' * len(self.data), *self.data)
        elif self.data_type == 'token':
            return bytes(self.data)
        raise ValueError("Unsupported data type.")

    def to_base64(self) -> str:
        return base64.b64encode(self.to_bytes()).decode()

    def to_hex(self) -> str:
        return self.to_bytes().hex()

    def __and__(self, other: "DataUnit") -> "DataUnit":
        if self.data_type == 'bytes' and other.data_type == 'bytes':
            value = bytes(a & b for a, b in zip(self.data, other.data))
            return DataUnit(value)
        raise TypeError("AND operation only supported for byte data.")

    def __or__(self, other: "DataUnit") -> "DataUnit":
        if self.data_type == 'bytes' and other.data_type == 'bytes':
            value = bytes(a | b for a, b in zip(self.data, other.data))
            return DataUnit(value)
        raise TypeError("OR operation only supported for byte data.")

    def to_dataclass(self):
        return DataUnitDataclass(self.data_type, self.data)


@dataclass(frozen=True)
class DataUnitDataclass(Atom):
    data_type: str
    data: Union[bytes, List[float], List[int]]

    def __post_init__(self):
        if self.data_type not in ['string', 'bytes', 'embedding', 'token']:
            raise ValueError("Unsupported data type.")

    def __repr__(self):
        if self.data_type == 'string':
            return f"DataUnitDataclass(string: {self.data.decode()})"
        elif self.data_type == 'bytes':
            return f"DataUnitDataclass(bytes: {self.data.hex()})"
        elif self.data_type == 'embedding':
            return f"DataUnitDataclass(embedding: {self.data})"
        elif self.data_type == 'token':
            return f"DataUnitDataclass(token: {self.data})"
        return "DataUnitDataclass(unsupported type)"

    def to_dataclass(self):
        return self


@dataclass(frozen=True)
class AtomDataclass(Generic[T], Atom):
    value: T

    def __repr__(self):
        return f"AtomDataclass(value={self.value})"

    def to_dataclass(self):
        return self

    def __add__(self, other: 'AtomDataclass[T]') -> 'AtomDataclass[T]':
        return AtomDataclass(operator.add(self.value, other.value))

    def __sub__(self, other: 'AtomDataclass[T]') -> 'AtomDataclass[T]':
        return AtomDataclass(operator.sub(self.value, other.value))

    def __mul__(self, other: 'AtomDataclass[T]') -> 'AtomDataclass[T]':
        return AtomDataclass(operator.mul(self.value, other.value))

    def __truediv__(self, other: 'AtomDataclass[T]') -> 'AtomDataclass[T]':
        return AtomDataclass(operator.truediv(self.value, other.value))

    def __pow__(self, other: 'AtomDataclass[T]') -> 'AtomDataclass[T]':
        return AtomDataclass(operator.pow(self.value, other.value))

    def __eq__(self, other: 'AtomDataclass[T]') -> bool:
        return operator.eq(self.value, other.value)

    def __ne__(self, other: 'AtomDataclass[T]') -> bool:
        return operator.ne(self.value, other.value)

    def __lt__(self, other: 'AtomDataclass[T]') -> bool:
        return operator.lt(self.value, other.value)

    def __gt__(self, other: 'AtomDataclass[T]') -> bool:
        return operator.gt(self.value, other.value)

    def __le__(self, other: 'AtomDataclass[T]') -> bool:
        return operator.le(self.value, other.value)

    def __ge__(self, other: 'AtomDataclass[T]') -> bool:
        return operator.ge(self.value, other.value)

    def __and__(self, other: 'AtomDataclass[T]') -> 'AtomDataclass[T]':
        return AtomDataclass(operator.and_(self.value, other.value))

    def __or__(self, other: 'AtomDataclass[T]') -> 'AtomDataclass[T]':
        return AtomDataclass(operator.or_(self.value, other.value))

    def __invert__(self) -> 'AtomDataclass[T]':
        return AtomDataclass(operator.not_(self.value))


# Formal Theory and Operations Classes

@dataclass
class FormalTheory(Generic[T]):
    """Represents a formal theory with basic properties and operations."""
    reflexivity: Callable[[T], bool] = operator.eq
    symmetry: Callable[[T, T], bool] = operator.eq
    transitivity: Callable[[T, T, T], bool] = lambda x, y, z: (x == y) and (y == z) and (x == z)
    transparency: Callable[[Callable[..., T], T, T], T] = lambda f, x, y: f(x) if x == y else None
    case_base: dict[str, Callable[[T, T], T]] = field(default_factory=dict)

    def __post_init__(self):
        self.case_base = {
            '⊤': lambda x, _: x,
            '⊥': lambda _, y: y,
            'a': partial(self.if_else, a=True)
        }

    def __str__(self):
        return f"FormalTheory({self.reflexivity.__name__}, {self.symmetry.__name__}, {self.transitivity.__name__}, {self.transparency.__name__}, {self.case_base})"

    def __repr__(self):
        case_base_str = {k: v.__name__ if hasattr(v, '__name__') else str(v) for k, v in self.case_base.items()}
        return (f"FormalTheory(\n"
                f"  reflexivity={self.reflexivity.__name__},\n"
                f"  symmetry={self.symmetry.__name__},\n"
                f"  transitivity={self.transitivity.__name__},\n"
                f"  transparency={self.transparency.__name__},\n"
                f"  case_base={case_base_str}\n"
                f")")

    @staticmethod
    def if_else(a: bool, x: T, y: T) -> T:
        return x if a else y


class Operations(ABC, Generic[T]):
    """An abstract base class for defining operations."""

    @abstractmethod
    def equality(self, x: T, y: T) -> bool:
        """Check if two elements are equal."""
        pass

    @abstractmethod
    def less_than_or_equal_to(self, x: T, y: T) -> bool:
        """Check if one element is less than or equal to another."""
        pass

    @abstractmethod
    def greater_than(self, x: T, y: T) -> bool:
        """Check if one element is greater than another."""
        pass

    @abstractmethod
    def negation(self, a: T) -> T:
        """Negate an element."""
        pass

    @abstractmethod
    def excluded_middle(self, a: T, b: T) -> T:
        """Apply the law of excluded middle."""
        pass

    @abstractmethod
    def and_(self, a: T, b: T) -> T:
        """Perform logical AND operation."""
        pass

    @abstractmethod
    def or_(self, a: T, b: T) -> T:
        """Perform logical OR operation."""
        pass

    @abstractmethod
    def implication(self, a: T, b: T) -> bool:
        """Check if one element implies another."""
        pass

    @abstractmethod
    def conjunction(self, *args: T) -> T:
        """Perform conjunction over a sequence of elements."""
        pass

    @abstractmethod
    def disjunction(self, *args: T) -> T:
        """Perform disjunction over a sequence of elements."""
        pass

    def inequality(self, x: T, y: T) -> bool:
        """Check if two elements are not equal."""
        return not self.equality(x, y)

    def less_than(self, x: T, y: T) -> bool:
        """Check if one element is less than another."""
        return self.greater_than(y, x)

    def greater_than_or_equal_to(self, x: T, y: T) -> bool:
        """Check if one element is greater than or equal to another."""
        return self.less_than_or_equal_to(y, x)

    def double_negation(self, a: T) -> T:
        """Apply double negation to an element."""
        return self.negation(self.negation(a))

    def biconditional(self, a: T, b: T) -> bool:
        """Check if two elements are biconditionally related."""
        return self.implication(a, b) and self.implication(b, a)


class DefaultOperations(Operations[Any]):
    """A default implementation of the Operations class using Python's built-in operators."""

    def equality(self, x: Any, y: Any) -> bool:
        return operator.eq(x, y)

    def less_than_or_equal_to(self, x: Any, y: Any) -> bool:
        return operator.le(x, y)

    def greater_than(self, x: Any, y: Any) -> bool:
        return operator.gt(x, y)

    def negation(self, a: Any) -> Any:
        return operator.not_(a)

    def excluded_middle(self, a: Any, b: Any) -> Any:
        return self.negation(self.and_(a, b)) or (self.negation(a) and self.negation(b))

    def and_(self, a: Any, b: Any) -> Any:
        return operator.and_(a, b)

    def or_(self, a: Any, b: Any) -> Any:
        return operator.or_(a, b)

    def implication(self, a: Any, b: Any) -> bool:
        return not a or b

    def conjunction(self, *args: Any) -> Any:
        return reduce(operator.and_, args)

    def disjunction(self, *args: Any) -> Any:
        return reduce(operator.or_, args)

    def __repr__(self):
        return ("DefaultOperations(\n"
                "  equality=operator.eq,\n"
                "  less_than_or_equal_to=operator.le,\n"
                "  greater_than=operator.gt,\n"
                "  negation=operator.not_,\n"
                "  excluded_middle=(lambda a, b: not (a & b) or (not a & not b)),\n"
                "  and_=operator.and_,\n"
                "  or_=operator.or_,\n"
                "  implication=(lambda a, b: not a or b),\n"
                "  conjunction=(lambda *args: reduce(operator.and_, args)),\n"
                "  disjunction=(lambda *args: reduce(operator.or_, args))\n"
                ")")

# Example usage
if __name__ == "__main__":
    formal_theory = FormalTheory()
    operations = DefaultOperations()

    x, y, z = 1, 2, 3
    print(operations)
    print(formal_theory)
    print(f'"Does x = y"?: {operations.equality(x, y)}')
    print(f'"Is y > x"?: {operations.greater_than(y, x)}')