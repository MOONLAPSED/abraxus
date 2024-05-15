from dataclasses import dataclass
from typing import Any, Callable, TypeVar, Generic
import operator
from functools import reduce
from .atom import Atom, DataUnit

T = TypeVar('T')

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

# Arithmetic Properties
def commutativity(op: Callable[[T, T], T], a: AtomDataclass[T], b: AtomDataclass[T]) -> bool:
    return op(a.value, b.value) == op(b.value, a.value)

def associativity(op: Callable[[T, T], T], a: AtomDataclass[T], b: AtomDataclass[T], c: AtomDataclass[T]) -> bool:
    return op(op(a.value, b.value), c.value) == op(a.value, op(b.value, c.value))

def distributivity(a: AtomDataclass[T], b: AtomDataclass[T], c: AtomDataclass[T]) -> bool:
    return (a * (b + c)).value == (a * b).value + (a * c).value

# Logical Foundations
def and_operator(a: AtomDataclass[bool], b: AtomDataclass[bool]) -> AtomDataclass[bool]:
    return a & b

def or_operator(a: AtomDataclass[bool], b: AtomDataclass[bool]) -> AtomDataclass[bool]:
    return a | b

def not_operator(a: AtomDataclass[bool]) -> AtomDataclass[bool]:
    return ~a

def implication(a: AtomDataclass[bool], b: AtomDataclass[bool]) -> AtomDataclass[bool]:
    return ~a | b

def biconditional(a: AtomDataclass[bool], b: AtomDataclass[bool]) -> AtomDataclass[bool]:
    return and_operator(implication(a, b), implication(b, a))

# Sets and Set Operations
set_operations = {
    'union': operator.or_,
    'intersection': operator.and_,
    'difference': operator.sub,
    'complement': operator.xor
}

# Functions and Relations
def function_application(f: Callable[[T], T], x: AtomDataclass[T]) -> AtomDataclass[T]:
    return AtomDataclass(f(x.value))

def composition(f: Callable[[T], T], g: Callable[[T], T]) -> Callable[[AtomDataclass[T]], AtomDataclass[T]]:
    def composed(x: AtomDataclass[T]) -> AtomDataclass[T]:
        return AtomDataclass(f(g(x.value)))
    return composed