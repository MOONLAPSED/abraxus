import code
import time
from functools import wraps
from dataclasses import dataclass, field

def temporal_mro_decorator(cls):
    """
    Decorator to enhance a class with logging and timing of method executions.
    Wraps each method and logs the time taken to execute with causal information.
    """
    methods = {}

    def wrapper(func):
        @wraps(func)
        def inner(self, *args, **kwargs):
            start_time = time.perf_counter()
            result = func(self, *args, **kwargs)
            end_time = time.perf_counter()
            duration = end_time - start_time
            method_name = func.__name__
            methods[method_name] = methods.get(method_name, []) + [(duration, args, kwargs)]
            self.log_method_execution(method_name, duration, args, kwargs)
            return result
        return inner

    for attr_name in dir(cls):
        attr_value = getattr(cls, attr_name)
        if callable(attr_value) and not attr_name.startswith("__"):
            setattr(cls, attr_name, wrapper(attr_value))

    return cls

@dataclass
class LoggingMixin:
    """
    Mixin class to provide logging functionality for decorated classes.
    """
    method_log: list = field(default_factory=list, init=False)

    def log_method_execution(self, method_name, duration, args, kwargs):
        log_entry = {
            "method_name": method_name,
            "duration": duration,
            "args": args,
            "kwargs": kwargs
        }
        self.method_log.append(log_entry)
        print(f"[LOG] {method_name} executed in {duration:.6f}s with args: {args} kwargs: {kwargs}")

class DynamicClassCreator:
    """
    Class responsible for dynamic creation of data classes in the REPL.
    """
    def __init__(self):
        self.namespace = {}

    def create_dynamic_class(self, class_name, fields):
        """
        Dynamically creates a new data class with the specified fields and applies the decorator.
        """
        @temporal_mro_decorator
        @dataclass
        class DynamicDataClass(LoggingMixin):
            pass

        for field_name in fields:
            setattr(DynamicDataClass, field_name, 0)

        self.namespace[class_name] = DynamicDataClass

    def interact(self):
        """
        Start the custom REPL session.
        """
        console = code.InteractiveConsole(self.namespace)
        console.interact("Dynamic REPL. Define classes and methods interactively.")

if __name__ == "__main__":
    creator = DynamicClassCreator()

    # Example: Create a class dynamically with field names.
    creator.create_dynamic_class("MyClass", ["x", "y", "z"])
    
    # Instantiate and work with the class via REPL
    # Example: Interact via REPL to create an instance and invoke methods
    creator.interact()