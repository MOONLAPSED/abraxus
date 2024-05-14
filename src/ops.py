# src/ops.py

import operator
from typing import Any, Callable, TypeVar
from functools import reduce
from .atom import Atom

T = TypeVar('T')

class DefaultOperations(Atom):
    """
    A default implementation of basic operations using Python's built-in operators.
    """
  
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

# Arithmetic Operations
def addition(a: T, b: T) -> T:
    return operator.add(a, b)

def subtraction(a: T, b: T) -> T:
    return operator.sub(a, b)

def multiplication(a: T, b: T) -> T:
    return operator.mul(a, b)

def division(a: T, b: T) -> T:
    return operator.truediv(a, b)

def exponentiation(a: T, b: T) -> T:
    return operator.pow(a, b)

# Arithmetic Properties
def commutativity(op: Callable[[T, T], T], a: T, b: T) -> bool:
    return op(a, b) == op(b, a)

def associativity(op: Callable[[T, T], T], a: T, b: T, c: T) -> bool:
    return op(op(a, b), c) == op(a, op(b, c))

def distributivity(a: T, b: T, c: T) -> bool:
    return multiplication(a, addition(b, c)) == addition(multiplication(a, b), multiplication(a, c))

# Ordering and Inequalities
def equality(a: T, b: T) -> bool:
    return operator.eq(a, b)

def inequality(a: T, b: T) -> bool:
    return operator.ne(a, b)

def less_than(a: T, b: T) -> bool:
    return operator.lt(a, b)

def greater_than(a: T, b: T) -> bool:
    return operator.gt(a, b)

def less_than_or_equal_to(a: T, b: T) -> bool:
    return operator.le(a, b)

def greater_than_or_equal_to(a: T, b: T) -> bool:
    return operator.ge(a, b)

def trichotomy(a: T, b: T) -> bool:
    return less_than(a, b) or equality(a, b) or greater_than(a, b)

# Logical Foundations
def and_operator(a: bool, b: bool) -> bool:
    return operator.and_(a, b)

def or_operator(a: bool, b: bool) -> bool:
    return operator.or_(a, b)

def not_operator(a: bool) -> bool:
    return operator.not_(a)

def implication(a: bool, b: bool) -> bool:
    return not_operator(a) or b

def biconditional(a: bool, b: bool) -> bool:
    return and_operator(implication(a, b), implication(b, a))

# Sets and Set Operations
set_operations = {
    'union': operator.or_,
    'intersection': operator.and_,
    'difference': operator.sub,
    'complement': operator.xor
}

# Functions and Relations
def function_application(f: Callable[[T], T], x: T) -> T:
    return f(x)

def composition(f: Callable[[T], T], g: Callable[[T], T]) -> Callable[[T], T]:
    def composed(x: T) -> T:
        return f(g(x))
    return composed