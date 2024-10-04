import inspect
import logging
from enum import Enum, auto
from typing import Any, Dict, List, Tuple, Union, Callable, TypeVar, Type
from functools import wraps
import sys
import pathlib
import asyncio
import time

"""Type Variables to allow type-checking, linting,.. of Generic...
    "T"((t)ypes and classes),
    "V"((v)ariables and functions),
    "C"((c)allable(reflective functions))"""
T = TypeVar('T', bound=Type)  # type is synonymous for class: T = type(class()) or vice-versa
V = TypeVar('V', bound=Union[int, float, str, bool, list, dict, tuple, set, object, Callable, Enum, Type[Any]])
C = TypeVar('C', bound=Callable[..., Any])  # callable 'T' class/type variable

# Data types
datum = Union[int, float, str, bool, None, List[Any], Tuple[Any, ...]]

class DataType(Enum):
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    BOOLEAN = auto()
    NONE = auto()
    LIST = auto()
    TUPLE = auto()

# Logging decorator
def _log(level=logging.INFO):
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logging.log(level, f"Executing {func.__name__} with args: {args}, kwargs: {kwargs}")
            result = await func(*args, **kwargs)
            logging.log(level, f"Completed {func.__name__} with result: {result}")
            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logging.log(level, f"Executing {func.__name__} with args: {args}, kwargs: {kwargs}")
            result = func(*args, **kwargs)
            logging.log(level, f"Completed {func.__name__} with result: {result}")
            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Benchmarking decorator
def _bench(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not getattr(sys, 'bench', True):  # Disable benchmark if sys.bench is False
            return await func(*args, **kwargs)
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        logging.info(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Introspection function
def _introspection(obj: Any, depth: int = 1):
    logging.info(f"Introspecting: {obj.__class__.__name__}")
    for name, value in inspect.getmembers(obj):
        if not name.startswith('_'):
            if inspect.isfunction(value) or inspect.ismethod(value):
                logging.info(f"{'  ' * depth}Method: {name}")
            elif isinstance(value, property):
                logging.info(f"{'  ' * depth}Property: {name}")
            else:
                logging.info(f"{'  ' * depth}Attribute: {name} = {value}")
                if isinstance(value, BaseModel) and depth < 3:  # Example for nested inspection
                    _introspection(value, depth + 1)

# Base Model
class BaseModel:
    __slots__ = ('__dict__', '__weakref__')

    def __init__(self, **data):
        for name, value in data.items():
            setattr(self, name, value)

    def __setattr__(self, name, value):
        if name in self.__annotations__:
            expected_type = self.__annotations__[name]
            if not isinstance(value, expected_type):
                raise TypeError(f"Expected {expected_type} for {name}, got {type(value)}")
            
            # Apply validation if defined
            validator = getattr(self.__class__, f'validate_{name}', None)
            if validator:
                validator(self, value)
        
        super().__setattr__(name, value)

    @classmethod
    def create(cls, **kwargs):
        return cls(**kwargs)

    def dict(self):
        return {name: getattr(self, name) for name in self.__annotations__}

    def __repr__(self):
        attrs = ', '.join(f"{name}={getattr(self, name)!r}" for name in self.__annotations__)
        return f"{self.__class__.__name__}({attrs})"

    def __str__(self):
        attrs = ', '.join(f"{name}={getattr(self, name)}" for name in self.__annotations__)
        return f"{self.__class__.__name__}({attrs})"
    
    def clone(self):  # singleton pattern
        return self.__class__(**self.dict())

def frozen(cls):  # Frozen Model decorator
    original_setattr = cls.__setattr__

    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError(f"Cannot modify frozen attribute '{name}'")
        original_setattr(self, name, value)
    
    cls.__setattr__ = __setattr__
    return cls

def validate(validator: Callable[[Any], None]):  # Validator decorator
    def decorator(func):
        @wraps(func)
        def wrapper(self, value):
            return validator(value)
        return wrapper
    return decorator

class FileModel(BaseModel):  # Dynamic Model Class Creation
    file_name: str
    file_content: str
    
    def save(self, directory: pathlib.Path):
        """Method to save content back to a file."""
        with (directory / self.file_name).open('w') as file:
            file.write(self.file_content)

@frozen
class Module(BaseModel):
    file_path: pathlib.Path
    module_name: str
    @validate(lambda x: x.endswith('.py'))
    def validate_file_path(self, value):
        return value
    @validate(lambda x: x.isidentifier())
    def validate_module_name(self, value):
        return value
    @frozen
    def __init__(self, file_path: pathlib.Path, module_name: str):
        super().__init__(file_path=file_path, module_name=module_name)
        self.file_path = file_path
        self.module_name = module_name
        self.file_content = None
    
    def __repr__(self):
        return f"Module(file_path={self.file_path}, module_name={self.module_name})"

    def __eq__(self, other):
        return self.file_path == other.file_path and self.module_name == other.module_name

def create_model_from_file(file_path: pathlib.Path):
    """Create a FileModel instance from a given file."""
    try:
        with file_path.open('r', encoding='utf-8', errors='ignore') as file:
            content = file.read()

        model_name = file_path.stem.capitalize() + 'Model'
        model_class = type(model_name, (FileModel,), {})

        instance = model_class.create(file_name=file_path.name, file_content=content)
        logging.info(f"Created {model_name} from {file_path}")

        return model_name, instance
    except Exception as e:
        logging.error(f"Failed to create model from {file_path}: {e}")
        return None, None

def load_files_as_models(root_dir: pathlib.Path, file_extensions: List[str]) -> Dict[str, BaseModel]:
    """Recursively scan directories and load files as models based on extensions."""
    models = {}
    for file_path in root_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix in file_extensions:
            model_name, instance = create_model_from_file(file_path)
            if model_name and instance:
                models[model_name] = instance
                # Registering the instance in sys.modules for access
                sys.modules[model_name] = instance

    return models

def register_models(models: Dict[str, BaseModel]):
    """Register models in the global namespace."""
    for model_name, instance in models.items():
        globals()[model_name] = instance
        logging.info(f"Registered {model_name} in the global namespace")

def runtime(root_dir: pathlib.Path):
    """Main runtime function."""
    file_models = load_files_as_models(root_dir, ['.md', '.txt'])
    register_models(file_models)

def main():  # Demonstrate access to models and their use as file objects
    import argparse
    parser = argparse.ArgumentParser(description="Process files and create models.")
    parser.add_argument("directory", type=str, help="Directory to scan for files.")
    parser.add_argument("--extensions", type=str, nargs="+", default=[".md", ".txt"],
                        help="File extensions to include.")
    args = parser.parse_args()

    root_dir = pathlib.Path(args.directory)
    file_models = load_files_as_models(root_dir, args.extensions)
    
    register_models(file_models)

    runtime(root_dir)  # Call runtime() here, passing root_dir

if __name__ == "__main__":
    main()
    # python ./src/mixins.py /path/to/your/directory --extensions .py .json
