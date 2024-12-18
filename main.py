from __future__ import annotations
import asyncio
import inspect
import json
import logging
import os
import pathlib
import struct
import sys
import pickle
import threading
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from queue import Queue, Empty
from datetime import datetime
from typing import (
    Any, Dict, Optional, Union, Callable, TypeVar, Protocol, 
    runtime_checkable, List, Generic, Set, Coroutine, Type, ClassVar
)
class FrameModel(ABC):
    """A frame model is a data structure that contains the data of a frame aka a chunk of text contained by dilimiters.
        Delimiters are defined as '---' and '\n' or its analogues (EOF) or <|in_end|> or "..." etc for the start and end of a frame respectively.)
        the frame model is a data structure that is independent of the source of the data.
        portability note: "dilimiters" are established by the type of encoding and the arbitrary writing-style of the source data. eg: ASCII
    """
    @abstractmethod
    def to_bytes(self) -> bytes:
        """Return the frame data as bytes."""
        pass


class AbstractDataModel(FrameModel, ABC):
    """A data model is a data structure that contains the data of a frame aka a chunk of text contained by dilimiters.
        It has abstract methods --> to str and --> to os.pipe() which are implemented by the concrete classes.
    """
    @abstractmethod
    def to_pipe(self, pipe) -> None:
        """Write the model to a named pipe."""
        pass

    @abstractmethod
    def to_str(self) -> str:
        """Return the frame data as a string representation."""
        pass


class SerialObject(AbstractDataModel, ABC):
    """SerialObject is an abstract class that defines the interface for serializable objects within the abstract data model.
        Inputs:
            AbstractDataModel: The base class for the SerialObject class

        Returns:
            SerialObject object
    
    """
    @abstractmethod
    def dict(self) -> dict:
        """Return a dictionary representation of the model."""
        pass

    @abstractmethod
    def json(self) -> str:
        """Return a JSON string representation of the model."""
        pass
@dataclass
class ConcreteSerialModel(SerialObject):
    """
    This concrete implementation of SerialObject ensures that instances can
    be used wherever a FrameModel, AbstractDataModel, or SerialObject is required,
    hence demonstrating polymorphism.
        Inputs:
            SerialObject: The base class for the ConcreteSerialModel class

        Returns:
            ConcreteSerialModel object        
    """

    name: str
    age: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_bytes(self) -> bytes:
        """Return the JSON representation as bytes."""
        return self.json().encode()

    def to_pipe(self, pipe) -> None:
        """
        Write the JSON representation of the model to a named pipe.
        TODO: actual implementation needed for communicating with the pipe.
        """
        pass

    def to_str(self) -> str:
        """Return the JSON representation as a string."""
        return self.json()

    def dict(self) -> dict:
        """Return a dictionary representation of the model."""
        return {
            "name": self.name,
            "age": self.age,
            "timestamp": self.timestamp.isoformat(),
        }

    def json(self) -> str:
        """Return a JSON representation of the model as a string."""
        return json.dumps(self.dict())
    
    def to_pipe(self, pipe_name) -> None:
        """Write the JSON representation of the model to a named pipe."""
        write_to_pipe(pipe_name, self.json())

@dataclass
class Atom(AbstractDataModel):
    """Monolithic Atom class that integrates with the middleware model."""
    name: str
    value: Any
    elements: List[Element] = field(default_factory=list)
    serializer: ConcreteSerialModel = None  # Optional serializer

    def to_bytes(self) -> bytes:
        """Return the Atom's data as bytes."""
        return self.json().encode()

    def to_pipe(self, pipe_name) -> None:
        """Write the Atom's data to a named pipe."""
        # write_to_pipe(pipe_name, self.json())
        pass

    def to_str(self) -> str:
        """Return the Atom's data as a string representation."""
        return self.json()

    def dict(self) -> dict:
        """Return a dictionary representation of the Atom."""
        return {
            "name": self.name,
            "value": self.value,
            "elements": [element.dict() for element in self.elements],
        }

    def json(self) -> str:
        """Return a JSON representation of the Atom."""
        return json.dumps(self.dict())

    def serialize(self):
        """Serialize the Atom using its serializer, if provided."""
        if self.serializer:
            return self.serializer.to_json()
        else:
            return self.json()

"""py objects are implemented as C structures.
typedef struct _object {
    Py_ssize_t ob_refcnt;
    PyTypeObject *ob_type;
} PyObject; """
# Everything in Python is an object, and every object has a type. The type of an object is a class. Even the
# type class itself is an instance of type. Functions defined within a class become method objects when
# accessed through an instance of the class
"""(3.13 std lib)Functions are instances of the function class
Methods are instances of the method class (which wraps functions)
Both function and method are subclasses of object
homoiconism dictates the need for a way to represent all Python constructs as first class citizen(fcc):
    (functions, classes, control structures, operations, primitive values)
nominative 'true OOP'(SmallTalk) and my specification demands code as data and value as logic, structure.
The Atom(), our polymorph of object and fcc-apparent at runtime, always represents the literal source code
    which makes up their logic and possess the ability to be stateful source code data structure. """
# Atom()(s) are a wrapper that can represent any Python object, including values, methods, functions, and classes.
class AtomType(Enum):
    VALUE = auto()  # implies all Atom()(s) are both their source code and the value generated by their source code (at runtime)
    FUNCTION = auto()  # are fcc along with object, class, method, etc are polymorphs
    CLASS = auto()
    MODULE = auto()  # python 'module' ie. packaging, eg: code as data runtime 'database'
"""Homoiconism dictates that, upon runtime validation, all objects are code and data.
To facilitate; we utilize first class functions and a static typing system.
This maps perfectly to the three aspects of nominative invariance:
    Identity preservation, T: Type structure (static)
    Content preservation, V: Value space (dynamic)
    Behavioral preservation, C: Computation space (transformative)
    [[T (Type) ←→ V (Value) ←→ C (Callable)]] == 'quantum infodynamics, a triparte element; our Atom()(s)'
    Meta-Language (High Level)
      ↓ [First Collapse - Compilation]
    Intermediate Form (Like a quantum superposition)
      ↓ [Second Collapse - Runtime]
    Executed State (Measured Reality)
What's conserved across these transformations:
    Nominative relationships
    Information content
    Causal structure
    Computational potential"""
T = TypeVar('T', bound=any) # T for TypeVar, V for ValueVar. Homoicons are T+V.
V = TypeVar('V', bound=Union[int, float, str, bool, list, dict, tuple, set, object, Callable, type])
C = TypeVar('C', bound=Callable[..., Any])  # callable 'T'/'V' first class function interface
DataType = Enum('DataType', 'INTEGER FLOAT STRING BOOLEAN NONE LIST TUPLE DICT') # 'T' vars (stdlib)

@runtime_checkable
class Atom(Protocol):
    """
    Structural typing protocol for Atoms.
    Defines the minimal interface that an Atom must implement.
    """
    id: str

def atom(cls: Type[{T, V, C}]) -> Type[{T, V, C}]: # homoicon decorator
    """Decorator to create a homoiconic atom."""
    original_init = cls.__init__
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, 'id'):
            self.id = hashlib.sha256(self.__class__.__name__.encode('utf-8')).hexdigest()

    cls.__init__ = new_init
    return cls

AtomType = TypeVar('AtomType', bound=Atom)
QuantumAtomState = Enum('QuantumAtomState', ['SUPERPOSITION', 'ENTANGLED', 'COLLAPSED', 'DECOHERENT'])
"""The type system forms the "boundary" theory
The runtime forms the "bulk" theory
The homoiconic property ensures they encode the same information
The holoiconic property enables:
    States as quantum superpositions
    Computations as measurements
    Types as boundary conditions
    Runtime as bulk geometry"""




@dataclass
class ErrorAtom(Atom):
    error_type: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    traceback: Optional[str] = None

    @classmethod
    def from_exception(cls, exception: Exception, context: Dict[str, Any] = None):
        return cls(
            error_type=type(exception).__name__,
            message=str(exception),
            context=context or {},
            traceback=traceback.format_exc()
        )

class EventBus(Atom):  # Pub/Sub homoiconic event bus
    def __init__(self, id: str = "event_bus"):
        super().__init__(id=id)
        self._subscribers: Dict[str, List[Callable[[str, Any], Coroutine[Any, Any, None]]]] = {}

    async def subscribe(self, event_type: str, handler: Callable[[str, Any], Coroutine[Any, Any, None]]) -> None:
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    async def unsubscribe(self, event_type: str, handler: Callable[[str, Any], Coroutine[Any, Any, None]]) -> None:
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(handler)

    async def publish(self, event_type: str, event_data: Any) -> None:
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                asyncio.create_task(handler(event_type, event_data))

    def encode(self) -> bytes:
        raise NotImplementedError("EventBus cannot be directly encoded")

    @classmethod
    def decode(cls, data: bytes) -> None:
        raise NotImplementedError("EventBus cannot be directly decoded")

# Centralized Error Handling
class ErrorHandler:
    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus or EventBus()

    async def handle(self, error_atom: ErrorAtom):
        """
        Central error handling method with flexible error management
        """
        try:
            # Publish to system error channel
            await self.event_bus.publish('system.error', error_atom)
            
            # Optional: Log to file or external system
            self._log_error(error_atom)
            
            # Optional: Perform additional error tracking or notification
            await self._notify_error(error_atom)
        
        except Exception as secondary_error:
            # Fallback error handling
            print(f"Critical error in error handling: {secondary_error}")
            print(f"Original error: {error_atom}")

    def _log_error(self, error_atom: ErrorAtom):
        """
        Optional method to log errors to file or external system
        """
        with open('system_errors.log', 'a') as log_file:
            log_file.write(f"{error_atom.error_type}: {error_atom.message}\n")
            if error_atom.traceback:
                log_file.write(f"Traceback: {error_atom.traceback}\n")

    async def _notify_error(self, error_atom: ErrorAtom):
        """
        Optional method for additional error notification
        """
        # Could integrate with external monitoring systems
        # Could send alerts via different channels
        pass

    def decorator(self, func):
        """
        Error handling decorator for both sync and async functions
        """
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_atom = ErrorAtom.from_exception(e, context={
                    'args': args,
                    'kwargs': kwargs,
                    'function': func.__name__
                })
                await self.handle(error_atom)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_atom = ErrorAtom.from_exception(e, context={
                    'args': args,
                    'kwargs': kwargs,
                    'function': func.__name__
                })
                asyncio.run(self.handle(error_atom))
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

# Global error handler instance
global_error_handler = ErrorHandler()

# Convenience decorator
def handle_errors(func):
    return global_error_handler.decorator(func)

EventBus = EventBus('EventBus')

class HoloiconicTransform(Generic[T, V, C]):
    @staticmethod
    def flip(value: V) -> C:
        """Transform value to computation (inside-out)"""
        return lambda: value

    @staticmethod
    def flop(computation: C) -> V:
        """Transform computation to value (outside-in)"""
        return computation()

# Custom logger setup
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    return logger

Logger = setup_logger(__name__)

def log(level=logging.INFO):
    def decorator(func):
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

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def bench(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not getattr(sys, 'bench', True):
            return await func(*args, **kwargs)
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            end_time = time.perf_counter()
            Logger.info(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
            return result
        except Exception as e:
            Logger.exception(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

class BaseModel:
    __slots__ = ('__dict__', '__weakref__')

    def __setattr__(self, name, value):
        if name in self.__annotations__:
            expected_type = self.__annotations__[name]
            # Handle generic types and Any
            if hasattr(expected_type, '__origin__'):
                # Skip validation for generic types
                pass
            elif expected_type != Any:
                if not isinstance(value, expected_type):
                    raise TypeError(f"Expected {expected_type} for {name}, got {type(value)}")
            validator = getattr(self.__class__, f'validate_{name}', None)
            if validator:
                validator(self, value)
        super().__setattr__(name, value)

    @classmethod
    def create(cls, **kwargs):
        return cls(**kwargs)

    def dict(self):
        return {name: getattr(self, name) for name in self.__annotations__}

    def json(self):
        return json.dumps(self.dict())

    @classmethod
    def from_json(cls, json_str):
        return cls(**json.loads(json_str))

    def __repr__(self):
        attrs = ', '.join(f"{name}={getattr(self, name)!r}" for name in self.__annotations__)
        return f"{self.__class__.__name__}({attrs})"

    def __str__(self):
        attrs = ', '.join(f"{name}={getattr(self, name)}" for name in self.__annotations__)
        return f"{self.__class__.__name__}({attrs})"
    
    def clone(self):
        return self.__class__(**self.dict())

def frozen(cls):
    original_setattr = cls.__setattr__

    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError(f"Cannot modify frozen attribute '{name}'")
        original_setattr(self, name, value)
    
    cls.__setattr__ = __setattr__
    return cls

def validate(validator: Callable[[Any], None]):
    def decorator(func):
        @wraps(func)
        def wrapper(self, value):
            validator(value)
            return func(self, value)
        return wrapper
    return decorator

class FileModel(BaseModel):
    file_name: str
    file_content: str
    
    @log()
    def save(self, directory: pathlib.Path):
        try:
            with (directory / self.file_name).open('w') as file:
                file.write(self.file_content)
            Logger.info(f"Saved file: {self.file_name}")
        except IOError as e:
            Logger.error(f"Failed to save file {self.file_name}: {str(e)}")
            raise

@frozen
class Module(BaseModel):
    file_path: pathlib.Path
    module_name: str

    @validate(lambda x: x.suffix == '.py')
    def validate_file_path(self, value):
        return value

    @validate(lambda x: x.isidentifier())
    def validate_module_name(self, value):
        return value

    def __init__(self, file_path: pathlib.Path, module_name: str):
        super().__init__(file_path=file_path, module_name=module_name)

@log()
def create_model_from_file(file_path: pathlib.Path):
    try:
        with file_path.open('r', encoding='utf-8', errors='ignore') as file:
            content = file.read()

        model_name = file_path.stem.capitalize() + 'Model'
        # Create a proper class with BaseModel inheritance
        model_dict = {
            'file_name': str,
            'file_content': str,
            '__annotations__': {'file_name': str, 'file_content': str}
        }
        model_class = type(model_name, (FileModel,), model_dict)
        instance = model_class(file_name=file_path.name, file_content=content)
        Logger.info(f"Created {model_name} from {file_path}")
        return model_name, instance
    except Exception as e:
        Logger.error(f"Failed to create model from {file_path}: {e}")
        return None, None

@log()
def load_files_as_models(root_dir: pathlib.Path, file_extensions: List[str]) -> Dict[str, BaseModel]:
    models = {}
    for file_path in root_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix in file_extensions:
            model_name, instance = create_model_from_file(file_path)
            if model_name and instance:
                models[model_name] = instance
                sys.modules[model_name] = instance
    return models

@log()
def register_models(models: Dict[str, BaseModel]):
    for model_name, instance in models.items():
        globals()[model_name] = instance
        Logger.info(f"Registered {model_name} in the global namespace")

@log()
def runtime(root_dir: pathlib.Path):
    file_models = load_files_as_models(root_dir, ['.md', '.txt', '.py'])
    register_models(file_models)

TypeMap = {
    int: DataType.INTEGER,
    float: DataType.FLOAT,
    str: DataType.STRING,
    bool: DataType.BOOLEAN,
    type(None): DataType.NONE,
    list: DataType.LIST,
    tuple: DataType.TUPLE,
    dict: DataType.DICT
}

def get_type(value: Any) -> Optional[DataType]:
    return TypeMap.get(type(value))

def validate_datum(value: Any) -> bool:
    return get_type(value) is not None

def process_datum(value: Any) -> str:
    dtype = get_type(value)
    return f"Processed {dtype.name}: {value}" if dtype else "Unknown data type"

def safe_process_input(value: Any) -> str:
    return "Invalid input type" if not validate_datum(value) else process_datum(value)

def encode(atom: 'Atom') -> bytes:
    data = {
        'tag': atom.tag,
        'value': atom.value,
        'children': [encode(child) for child in atom.children],
        'metadata': atom.metadata
    }
    return pickle.dumps(data)

def decode(data: bytes) -> 'Atom':
    data = pickle.loads(data)
    atom = Atom(data['tag'], data['value'], [decode(child) for child in data['children']], data['metadata'])
    return atom

@dataclass
class Atom(BaseModel):
    id: str
    value: Any
    data_type: str = field(init=False)
    attributes: Dict[str, Any] = field(default_factory=dict)
    subscribers: Set['Atom'] = field(default_factory=set)
    MAX_INT_BIT_LENGTH: ClassVar[int] = 1024

    def __post_init__(self):
        self.data_type = self.infer_data_type(self.value)
        Logger.debug(f"Initialized Atom with value: {self.value} and inferred type: {self.data_type}")

    async def execute(self, *args, **kwargs) -> Any:
        Logger.debug(f"Atom {self.id} executing with value: {self.value}")
        return self.value

    def infer_data_type(self, value: Any) -> str:
        type_map = {
            str: 'string',
            int: 'integer',
            float: 'float',
            bool: 'boolean',
            list: 'list',
            dict: 'dictionary',
            type(None): 'none'
        }
        inferred_type = type_map.get(type(value), 'unsupported')
        Logger.debug(f"Inferred data type: {inferred_type}")
        return inferred_type

    @abstractmethod
    def encode(self) -> bytes:
        pass

    def validate(self) -> bool:
        return True

    def __getitem__(self, key: str) -> Any:
        return self.attributes[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def __delitem__(self, key: str) -> None:
        del self.attributes[key]

    def __contains__(self, key: str) -> bool:
        return key in self.attributes

    @log()
    async def send_message(self, message: Any, ttl: int = 3) -> None:
        if ttl <= 0:
            Logger.info(f"Message {message} dropped due to TTL")
            return
        Logger.info(f"Atom {self.id} received message: {message}")
        for sub in self.subscribers:
            await sub.receive_message(message, ttl - 1)

    @log()
    async def receive_message(self, message: Any, ttl: int) -> None:
        Logger.info(f"Atom {self.id} processing received message: {message} with TTL {ttl}")
        await self.send_message(message, ttl)

    def subscribe(self, atom: 'Atom') -> None:
        self.subscribers.add(atom)
        Logger.info(f"Atom {self.id} subscribed to {atom.id}")

    def unsubscribe(self, atom: 'Atom') -> None:
        self.subscribers.discard(atom)
        Logger.info(f"Atom {self.id} unsubscribed from {atom.id}")

class AntiAtom(Atom):
    def __init__(self, atom: Atom):
        super().__init__(id=f"anti_{atom.id}", value=None, attributes=atom.attributes)
        self.original_atom = atom

    def encode(self) -> bytes:
        return b'anti_' + self.original_atom.encode()

    async def execute(self, *args, **kwargs) -> Any:
        # Properly await the original atom's execute method
        result = await self.original_atom.execute(*args, **kwargs)
        return not result

@dataclass
class AtomicTheory:
    base_atom: Atom
    elements: List[Atom]
    theory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operations: Dict[str, Callable[..., Any]] = field(default_factory=lambda: {
        '⊤': lambda x: True,
        '⊥': lambda x: False,
        '¬': lambda x: not x,
        '∧': lambda a, b: a and b,
        '∨': lambda a, b: a or b,
        '→': lambda a, b: (not a) or b,
        '↔': lambda a, b: (a and b) or (not a and not b)
    })
    anti_theory: 'AntiAtom' = field(init=False)

    def __repr__(self) -> str:
        return f"AtomicTheory(theory_id={self.theory_id}, elements={self.elements!r}, operations={list(self.operations.keys())})"

    async def execute(self, operation: str, *args, **kwargs) -> Any:
        Logger.debug(f"Executing AtomicTheory {self.theory_id} operation: {operation} with args: {args}")
        if operation in self.operations:
            result = self.operations[operation](*args)
            Logger.debug(f"Operation result: {result}")
            return result
        else:
            raise ValueError(f"Operation {operation} not supported in AtomicTheory.")

    def __post_init__(self):
        self.anti_theory = AntiAtom(self.base_atom)
        Logger.debug(f"Initialized AtomicTheory with elements: {self.elements}")

    def __repr__(self) -> str:
        return f"AtomicTheory(theory_id={self.theory_id}, elements={self.elements!r}, operations={list(self.operations.keys())})"

    def add_operation(self, name: str, operation: Callable[..., Any]) -> None:
        Logger.debug(f"Adding operation '{name}' to AtomicTheory")
        self.operations[name] = operation

    def encode(self) -> bytes:
        Logger.debug("Encoding AtomicTheory")
        encoded_elements = b''.join([element.encode() for element in self.elements])
        return struct.pack(f'{len(encoded_elements)}s', encoded_elements)

    def decode(self, data: bytes) -> None:
        Logger.debug("Decoding AtomicTheory from bytes")
        # Splitting data for elements is dependent on specific encoding scheme, here simplified
        num_elements = struct.unpack('i', data[:4])[0]
        element_size = len(data[4:]) // num_elements
        segments = [data[4+element_size*i:4+element_size*(i+1)] for i in range(num_elements)]
        for element, segment in zip(self.elements, segments):
            element.decode(segment)
        Logger.debug(f"Decoded AtomicTheory elements: {self.elements}")

@dataclass
class TaskAtom(Atom): # Tasks are atoms that represent asynchronous potential actions
    task_id: int
    atom: Atom
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Base class field
    value: Any = None  # Base class field
    data_type: str = field(init=False)  # Base class field
    attributes: Dict[str, Any] = field(default_factory=dict)  # Base class field
    subscribers: Set['Atom'] = field(default_factory=set)  # Base class field

    async def run(self) -> Any:
        logging.info(f"Running task {self.task_id}")
        try:
            self.result = await self.atom.execute(*self.args, **self.kwargs)
            logging.info(f"Task {self.task_id} completed with result: {self.result}")
        except Exception as e:
            logging.error(f"Task {self.task_id} failed with error: {e}")
        return self.result

    def encode(self) -> bytes:
        return json.dumps(self.to_dict()).encode()

    @classmethod
    def decode(cls, data: bytes) -> 'TaskAtom':
        obj = json.loads(data.decode())
        return cls.from_dict(obj)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'atom': self.atom.to_dict(),
            'args': self.args,
            'kwargs': self.kwargs,
            'result': self.result
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskAtom':
        return cls(
            task_id=data['task_id'],
            atom=Atom.from_dict(data['atom']),
            args=tuple(data['args']),
            kwargs=data['kwargs'],
            result=data['result']
        )

class ArenaAtom(Atom):  # Arenas are threaded virtual memory Atoms appropriately-scoped when invoked
    def __init__(self, name: str):
        super().__init__(id=name)
        self.name = name
        self.local_data: Dict[str, Any] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor()
        self.running = False
        self.lock = threading.Lock()
    
    async def allocate(self, key: str, value: Any) -> None:
        with self.lock:
            self.local_data[key] = value
            logging.info(f"Arena {self.name}: Allocated {key} = {value}")
    
    async def deallocate(self, key: str) -> None:
        with self.lock:
            value = self.local_data.pop(key, None)
            logging.info(f"Arena {self.name}: Deallocated {key}, value was {value}")
    
    def get(self, key: str) -> Any:
        return self.local_data.get(key)
    
    def encode(self) -> bytes:
        data = {
            'name': self.name,
            'local_data': {key: value.to_dict() if isinstance(value, Atom) else value 
                           for key, value in self.local_data.items()}
        }
        return json.dumps(data).encode()

    @classmethod
    def decode(cls, data: bytes) -> 'ArenaAtom':
        obj = json.loads(data.decode())
        instance = cls(obj['name'])
        instance.local_data = {key: Atom.from_dict(value) if isinstance(value, dict) else value 
                               for key, value in obj['local_data'].items()}
        return instance
    
    async def submit_task(self, atom: Atom, args=(), kwargs=None) -> int:
        task_id = uuid.uuid4().int
        task = TaskAtom(task_id, atom, args, kwargs or {})
        await self.task_queue.put(task)
        logging.info(f"Submitted task {task_id}")
        return task_id
    
    async def task_notification(self, task: TaskAtom) -> None:
        notification_atom = AtomNotification(f"Task {task.task_id} completed")
        await self.send_message(notification_atom)
    
    async def run(self) -> None:
        self.running = True
        asyncio.create_task(self._worker())
        logging.info(f"Arena {self.name} is running")

    async def stop(self) -> None:
        self.running = False
        self.executor.shutdown(wait=True)
        logging.info(f"Arena {self.name} has stopped")
    
    async def _worker(self) -> None:
        while self.running:
            try:
                task: TaskAtom = await asyncio.wait_for(self.task_queue.get(), timeout=1)
                logging.info(f"Worker in {self.name} picked up task {task.task_id}")
                await self.allocate(f"current_task_{task.task_id}", task)
                await task.run()
                await self.task_notification(task)
                await self.deallocate(f"current_task_{task.task_id}")
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Error in worker: {e}")

@dataclass
class AtomNotification(Atom):  # nominative async message passing interface
    message: str

    def encode(self) -> bytes:
        return json.dumps({'message': self.message}).encode()

    @classmethod
    def decode(cls, data: bytes) -> 'AtomNotification':
        obj = json.loads(data.decode())
        return cls(message=obj['message'])

class EventBus(Atom):  # Pub/Sub homoiconic event bus
    def __init__(self):
        super().__init__(id="event_bus")
        self._subscribers: Dict[str, List[Callable[[Atom], Coroutine[Any, Any, None]]]] = {}

    async def subscribe(self, event_type: str, handler: Callable[[Atom], Coroutine[Any, Any, None]]) -> None:
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    async def unsubscribe(self, event_type: str, handler: Callable[[Atom], Coroutine[Any, Any, None]]) -> None:
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(handler)

    async def publish(self, event_type: str, event_data: Any) -> None:
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                asyncio.create_task(handler(event_type, event_data))


    def encode(self) -> bytes:
        raise NotImplementedError("EventBus cannot be directly encoded")

    @classmethod
    def decode(cls, data: bytes) -> None:
        raise NotImplementedError("EventBus cannot be directly decoded")

@dataclass
class EventAtom(Atom):  # Events are network-friendly Atoms, associates with a type and an id (USER-scoped), think; datagram
    id: str
    type: str
    detail_type: Optional[str] = None
    message: Union[str, List[Dict[str, Any]]] = field(default_factory=list)
    source: Optional[str] = None
    target: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def encode(self) -> bytes:
        return json.dumps(self.to_dict()).encode()

    @classmethod
    def decode(cls, data: bytes) -> 'EventAtom':
        obj = json.loads(data.decode())
        return cls.from_dict(obj)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "detail_type": self.detail_type,
            "message": self.message,
            "source": self.source,
            "target": self.target,
            "content": self.content,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventAtom':
        return cls(
            id=data["id"],
            type=data["type"],
            detail_type=data.get("detail_type"),
            message=data.get("message"),
            source=data.get("source"),
            target=data.get("target"),
            content=data.get("content"),
            metadata=data.get("metadata", {})
        )

    def validate(self) -> bool:
        required_fields = ['id', 'type']
        for field in required_fields:
            if not getattr(self, field):
                raise ValueError(f"Missing required field: {field}")
        return True

@dataclass
class ActionRequestAtom(Atom):  # User-initiated action request
    action: str
    params: Dict[str, Any]
    self_info: Dict[str, Any]
    echo: Optional[str] = None

    def encode(self) -> bytes:
        return json.dumps(self.to_dict()).encode()

    @classmethod
    def decode(cls, data: bytes) -> 'ActionRequestAtom':
        obj = json.loads(data.decode())
        return cls.from_dict(obj)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "params": self.params,
            "self_info": self.self_info,
            "echo": self.echo
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionRequestAtom':
        return cls(
            action=data["action"],
            params=data["params"],
            self_info=data["self_info"],
            echo=data.get("echo")
        )



















# Main application logic
async def run_atomic_theory_demo():
    base_atom = Atom(id="theory_base", value=None)
    theory = AtomicTheory(
        base_atom=base_atom,
        elements=[
            Atom(id="a", value=True),
            Atom(id="b", value=False)
        ],
        theory_id="demo_theory"  # Using theory_id here
    )
    print(theory)
    print(theory.anti_theory)

    result = await theory.execute('∧', theory.elements[0].value, theory.elements[1].value)
    print(f"Result of A ∧ B: {result}")

    result = await theory.anti_theory.execute('∧', theory.elements[0].value, theory.elements[1].value)
    print(f"Result of ¬(A ∧ B): {result}")

@log()
def main():
    root_dir = pathlib.Path("./")  # Adjust this path as needed
    file_models = load_files_as_models(root_dir, ['.md', '.txt']) # ['.md', '.txt', '.py']
    register_models(file_models)
    runtime(root_dir)

    asyncio.run(run_atomic_theory_demo())

if __name__ == "__main__":
    main()
    # Potential user interaction
    atom = Atom(id="user_input", value=None)
    transformed = HoloiconicTransform.flip(atom.value)
    original = HoloiconicTransform.flop(transformed)
    assert original == atom.value

    # Quick start
    from working.mro import LogicalMROExample, set_process_priority

    # Set process to low priority
    set_process_priority(0)

    # Analyze class structures
    analyzer = LogicalMROExample()
    result = analyzer.analyze_classes()
    print(result['s_expressions'])