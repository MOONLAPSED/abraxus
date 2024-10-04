import logging
import os
import sys
import importlib
import pathlib
import asyncio
import uuid
import json
import struct
import hashlib
import pickle
import inspect
import threading
import tracemalloc
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Tuple, Generic, Coroutine, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from asyncio import Queue as AsyncQueue
from functools import wraps

# Setup logging
tracemalloc.start()
IS_POSIX = os.name == 'posix'

class CustomFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: "\x1b[38;20m%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)\x1b[0m",
        logging.INFO: "\x1b[32;20m%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)\x1b[0m",
        logging.WARNING: "\x1b[33;20m%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)\x1b[0m",
        logging.ERROR: "\x1b[31;20m%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)\x1b[0m",
        logging.CRITICAL: "\x1b[31;1m%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)\x1b[0m",
    }
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self._fmt)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logger(name: str, level: int = logging.INFO, handlers: Optional[List[logging.Handler]] = None):
    if handlers is None:
        handlers = [logging.StreamHandler()]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()
    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(CustomFormatter())
        logger.addHandler(handler)
    return logger

# Typing
T = TypeVar('T')
V = TypeVar('V', bound=Union[int, float, str, bool, list, dict, tuple, set, object, Callable, type])
C = TypeVar('C', bound=Callable[..., Any])

# Data and Atom Types
class DataType(Enum):
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    BOOLEAN = auto()
    NONE = auto()
    LIST = auto()
    TUPLE = auto()

class AtomType(Enum):
    FUNCTION = auto()
    CLASS = auto()
    MODULE = auto()
    OBJECT = auto()

# Decorators
def calloc(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        tracemalloc.stop()
        return result
    return wrapper

def atom(cls: Type[Union[T, V, C]]) -> Type[Union[T, V, C]]:
    """Decorator to create a homoiconic atom."""
    original_init = cls.__init__
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, 'id'):
            self.id = hashlib.sha256(self.__class__.__name__.encode('utf-8')).hexdigest()
    cls.__init__ = new_init
    return cls

def log(level=logging.INFO):
    def decorator(func: Callable):
        is_coroutine = asyncio.iscoroutinefunction(func)
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            Logger.log(level, f"Executing {func.__name__} with args: {args}, kwargs: {kwargs}")
            try:
                result = await func(*args, **kwargs)
                Logger.log(level, f"Completed {func.__name__} with result: {result}")
                return result
            except Exception as e:
                Logger.exception(f"Error in {func.__name__}: {str(e)}")
                raise
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            Logger.log(level, f"Executing {func.__name__} with args: {args}, kwargs: {kwargs}")
            try:
                result = func(*args, **kwargs)
                Logger.log(level, f"Completed {func.__name__} with result: {result}")
                return result
            except Exception as e:
                Logger.exception(f"Error in {func.__name__}: {str(e)}")
                raise
        return async_wrapper if is_coroutine else sync_wrapper
    return decorator

def validate(cls: Type[T]) -> Type[T]:
    original_init = cls.__init__
    sig = inspect.signature(original_init)
    def new_init(self: T, *args: Any, **kwargs: Any) -> None:
        bound_args = sig.bind(self, *args, **kwargs)
        for key, value in bound_args.arguments.items():
            if key in cls.__annotations__:
                expected_type = cls.__annotations__.get(key)
                if not isinstance(value, expected_type):
                    raise TypeError(f"Expected {expected_type} for {key}, got {type(value)}")
        original_init(self, *args, **kwargs)
    cls.__init__ = new_init
    return cls

# Encoding and Decoding Functions
def encode(atom: 'Atom') -> bytes:
    return pickle.dumps(atom.to_dict())

def decode(data: bytes) -> 'Atom':
    return Atom.from_dict(pickle.loads(data))

# Abstract Class Atom and its Subclasses
@atom
@validate
@dataclass
class Atom(ABC):
    id: str = field(init=False)
    tag: str = ''
    children: List['Atom'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    reflexivity: Callable[[T], bool] = lambda x: x == x
    symmetry: Callable[[T, T], bool] = lambda x, y: x == y
    transitivity: Callable[[T, T, T], bool] = lambda x, y, z: (x == y and y == z)
    transparency: Callable[[Callable[..., T], T, T], T] = lambda f, x, y: f(True, x, y) if x == y else None
    case_base: Dict[str, Callable[..., bool]] = field(default_factory=dict, init=False)
    def __post_init__(self):
        self.case_base = {
            '⊤': lambda x, _: x,
            '⊥': lambda _, y: y,
            '¬': lambda a: not a,
            '∧': lambda a, b: a and b,
            '∨': lambda a, b: a or b,
            '→': lambda a, b: (not a) or b,
            '↔': lambda a, b: (a and b) or (not a and not b),
        }
    def encode(self) -> bytes:
        return json.dumps({
            'id': self.id,
            'tag': self.tag,
            'children': [child.encode() for child in self.children],
            'metadata': self.metadata,
        }).encode()

    @classmethod
    def decode(cls, data: bytes) -> 'Atom':
        decoded_data = json.loads(data.decode())
        children = [cls.decode(child) for child in decoded_data['children']]
        return cls(id=decoded_data['id'], tag=decoded_data['tag'], children=children, metadata=decoded_data['metadata'])

    def validate(self) -> bool:
        return True

    def __getitem__(self, key: str) -> Any:
        return self.metadata[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def __delitem__(self, key: str) -> None:
        del self.metadata[key]

    def __contains__(self, key: str) -> bool:
        return key in self.metadata

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, metadata={self.metadata})"

    async def send_message(self, message: Any, ttl: int = 3) -> None:
        if ttl <= 0:
            logging.info(f"Message {message} dropped due to TTL")
            return
        logging.info(f"Atom {self.id} received message: {message}")
        for sub in self.children:
            await sub.receive_message(message, ttl - 1)

    async def receive_message(self, message: Any, ttl: int) -> None:
        logging.info(f"Atom {self.id} processing received message: {message} with TTL {ttl}")
        await self.send_message(message, ttl)

    def subscribe(self, atom: 'Atom') -> None:
        self.children.append(atom)
        logging.info(f"Atom {self.id} subscribed to {atom.id}")

    def unsubscribe(self, atom: 'Atom') -> None:
        self.children.remove(atom)
        logging.info(f"Atom {self.id} unsubscribed from {atom.id}")

# Other Atom and Atom-derived classes
class AntiAtom(Atom):
    original_atom: Atom
    def encode(self) -> bytes:
        return b'anti_' + self.original_atom.encode()

    def execute(self, *args, **kwargs) -> Any:
        return not self.original_atom.execute(*args, **kwargs)

class LiteralAtom(Atom):
    async def evaluate(self) -> Any:
        if self.tag == 'add':
            results = await asyncio.gather(*(child.evaluate() for child in self.children))
            return sum(results)
        elif self.tag == 'negate':
            return -await self.children[0].evaluate()
        else:
            raise NotImplementedError(f"Evaluation not implemented for tag: {self.tag}")

class FileAtom(Atom):
    file_path: Path
    file_content: str = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.file_content = self._read_file(self.file_path)

    def _read_file(self, file_path: Path) -> str:
        with file_path.open('r', encoding='utf-8', errors='ignore') as file:
            return file.read()

    async def evaluate(self):
        return self.file_content

# Function to create and load models
def create_model_from_file(file_path: pathlib.Path) -> Tuple[Optional[str], Optional[FileAtom]]:
    try:
        with file_path.open('r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
        model_name = file_path.stem.capitalize() + 'Model'
        model_class = type(model_name, (FileAtom,), {})
        instance = model_class(file_path=file_path, file_content=content)
        logging.info(f"Created {model_name} from {file_path}")
        return model_name, instance
    except Exception as e:
        logging.error(f"Failed to create model from {file_path}: {e}")
        return None, None

def load_files_as_models(root_dir: pathlib.Path, file_extensions: List[str]) -> Dict[str, Atom]:
    models = {}
    for file_path in root_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix in file_extensions:
            model_name, instance = create_model_from_file(file_path)
            if model_name and instance:
                models[model_name] = instance
                sys.modules[model_name] = instance
    return models

def register_models(models: Dict[str, Atom]) -> None:
    for model_name, instance in models.items():
        globals()[model_name] = instance
        logging.info(f"Registered {model_name} in the global namespace")

def runtime(root_dir: pathlib.Path) -> None:
    file_models = load_files_as_models(root_dir, ['.md', '.txt'])
    register_models(file_models)