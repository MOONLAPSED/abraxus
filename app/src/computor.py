#!/usr/bin/env python
# -*- coding: utf-8 -*-
# installed via pdm on either platform
import os
import time
import uuid
import logging
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, Any, List, Callable, Union, Type, TypeVar
from dataclasses import dataclass, field
"""
ganymede/
│
├── src/
│   ├── __init__.py
│   ├── app.py
│
└── main.py
"""


__all__ = []
"""py objects are implemented as C structures.
typedef struct _object {
    Py_ssize_t ob_refcnt;
    PyTypeObject *ob_type;
} PyObject; """
# Everything in Python is an object, and every object has a type. The type of an object is a class. Even the type class itself is an instance of type.
T = TypeVar('T', bound=Type)  # type is synonymous for class: T = type(class()) or vice-versa; are still ffc function
# functions defined within a class become method objects when accessed through an instance of the class
V = TypeVar('V', bound=Union[int, float, str, bool, list, dict, tuple, set, object, Callable, Enum, Type[Any]])  # Value variables
"""(3.12 std lib)Functions are instances of the function class
Methods are instances of the method class (which wraps functions)
Both function and method are subclasses of object
homoiconism dictates the need for a way to represent all Python constructs as first class citizen(fcc):
    (functions, classes, control structures, operations, primitive values)
nominative 'true OOP'(SmallTalk) and my specification demands code as data and value as logic, structure.
The Atom(), our polymorph of object and fcc-apparent at runtime, always represents the literal source code
    which makes up their logic and posess the ability to be stateful source code data structure. """
# Atom()(s) are a wrapper that can represent any Python object, including values, methods, functions, and classes.
class AtomType(Enum):
    VALUE = auto()  # implies all Atom()(s) are both their source code and the value generated by their source code (at runtime)
    FUNCTION = auto()  # are fcc along with object, class, method, etc are polymorphs
    CLASS = auto()
    MODULE = auto()  # python 'module' ie. packaging, eg: code as data runtime 'database'
# Logger setup
class CustomFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: "\x1b[38;20m%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)\x1b[0m",
        logging.INFO: "\x1b[32;20m%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)\x1b[0m",
        logging.WARNING: "\x1b[33;20m%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)\x1b[0m",
        logging.ERROR: "\x1b[31;20m%(asctime)s - %(name)s - %(levellevel)s - %(message)s (%(filename)s:%(lineno)d)\x1b[0m",
        logging.CRITICAL: "\x1b[31;1m%(asctime)s - %(name)s - %(levellevel)s - %(message)s (%(filename)s:%(lineno)d)\x1b[0m",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self._fmt)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(CustomFormatter())
    logger.addHandler(handler)
    return logger

logger = setup_logger('root', logging.DEBUG)

# Typing and Enums ----------------------------------------------------------
DataType = Enum('DataType', 'INTEGER FLOAT STRING BOOLEAN NONE LIST TUPLE')
AtomType = Enum('AtomType', 'FUNCTION CLASS MODULE OBJECT')

# Abstract Base Classes
class ISerializable(ABC):
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        pass

    @abstractmethod
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Deserialize from dictionary."""
        pass

@dataclass
class Atom(ISerializable):
    name: str
    tag: str
    is_callable: bool
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tag": self.tag,
            "is_callable": self.is_callable,
            "metadata": self.metadata
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        self.name = data["name"]
        self.tag = data["tag"]
        self.is_callable = data["is_callable"]
        self.metadata = data["metadata"]

@dataclass
class Constraints:
    B1_SYMBOLIC_CONFIGURATION = 256
    B2_INTERNAL_STATES = 1024
    L1_CONFIGURATION_CHANGE = 1
    D1_NEXT_COMP_STEP = 1

class Computor(ABC):
    def __init__(self, name: str, config: Dict[str, Any], logger: logging.Logger):
        self.name = name
        self.config = config
        self.logger = logger
        self.logger.info(f"Computor {self.name} initialized with config: {self.config}")

    @abstractmethod
    def execute(self, *args, **kwargs):
        pass

class ConcreteComputor(Computor):
    def __init__(self, name: str, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(name, config, logger)

    def execute(self, *args, **kwargs):
        self.logger.info(f"{self.name}: Executing with args: {args}, kwargs: {kwargs}")
        return sum(args) if args else 0

class AntiComputor(Computor, Constraints):
    def __init__(self, name: str, config: Dict[str, Any], logger: logging.Logger, is_guard: bool, is_callback: bool):
        super().__init__(name, config, logger)
        self.is_guard = is_guard
        self.is_callback = is_callback
        self.anti_anti_guard_callback = False

    def execute(self, *args, **kwargs):
        self.logger.info(f"{self.name} (AntiComputor): Executing with constraints")
        result = super().execute(*args, **kwargs)
        if self.is_guard:
            self.logger.info(f"{self.name} is guarding the execution")
        if self.is_callback:
            self.logger.info(f"{self.name} is providing a callback")
        return result

# Benchmarking Utilities
def benchmark(func: Callable, *args, **kwargs) -> float:
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    duration = end_time - start_time
    return duration, result

def main():
    # Create Computors
    computor1 = ConcreteComputor(name="Comp1", config={"param": 1}, logger=logger)
    anti_computor1 = AntiComputor(name="AntiComp1", config={"param": 1}, logger=logger, is_guard=True, is_callback=True)
    
    # Benchmarking Computors
    args = [1, 2, 3]
    duration, result = benchmark(computor1.execute, *args)
    logger.info(f"ConcreteComputor executed in {duration:.6f}s with result: {result}")

    duration, result = benchmark(anti_computor1.execute, *args)
    logger.info(f"AntiComputor executed in {duration:.6f}s with result: {result}")

if __name__ == "__main__":
    main()