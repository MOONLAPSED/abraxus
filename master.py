#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------#####PLATFORM&LOGGING###########-------------------------------#
# platforms: Ubuntu-22.04LTS (posix), Windows-11 (nt)
import asyncio
import inspect
import json
import logging
import os
import hashlib
import platform
import pathlib
import struct
import sys
import threading
import time
import shlex
import shutil
import uuid
import pickle

import argparse
import functools
from functools import wraps, lru_cache
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Union, Callable, TypeVar, Tuple, Generic, Set, Coroutine,
    Type, NamedTuple, ClassVar, Protocol
    )
from types import SimpleNamespace
from queue import Queue, Empty
from asyncio import Queue as AsyncQueue
import ctypes
import ast
import io
import importlib as _importlib
from importlib.util import spec_from_file_location, module_from_spec
import re
import dis
import linecache
import tracemalloc
# ----------------non-homoiconic pre-runtime "ADMIN-SCOPED" source code-------------------------#
IS_WINDOWS = os.name == 'nt'
IS_POSIX = os.name == 'posix'
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;20m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.format)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_queue = Queue()
        self.log_thread = threading.Thread(target=self.log_thread_func)
        self.log_thread.start()
    def log_thread_func(self):
        while True:
            try:
                record = self.log_queue.get()
                if record is None:
                    break
                self.handle(record)
            except Exception:
                import traceback
                print("Error in log thread:", file=sys.stderr)
                traceback.print_exc()
    def emit(self, record):
        self.log_queue.put(record)
    def close(self):
        self.log_queue.put(None)
        self.log_thread.join()
    def AdminLogger(self, name=None):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(self)
        logger.addHandler(ch)
        return logger
class AdminLogger(logging.LoggerAdapter):
    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})
    def process(self, msg, kwargs):
        return f"{self.extra['name']}: {msg}", kwargs
logger = AdminLogger(logging.getLogger(__name__))
# STATIC TYPING ========================================================
"""Homoiconism dictates that, upon runtime validation, all objects are code and data.
To fascilitate; we utilize first class functions and a static typing system."""
T = TypeVar('T', bound=any) # T for TypeVar, V for ValueVar. Homoicons are T+V.
V = TypeVar('V', bound=Union[int, float, str, bool, list, dict, tuple, set, object, Callable, type])
C = TypeVar('C', bound=Callable[..., Any])  # callable 'T'/'V' first class function interface
DataType = Enum('DataType', 'INTEGER FLOAT STRING BOOLEAN NONE LIST TUPLE') # 'T' vars (stdlib)
AtomType = Enum('AtomType', 'FUNCTION CLASS MODULE OBJECT') # 'C' vars (homoiconic methods or classes)
# HOMOICONISTIC morphological source code displays 'modified quine' behavior
# within a validated runtime, if and only if the valid python interpreter
# has r/w/x permissions to the source code file and some method of writing
# state to the source code file is available. Any interruption of the
# '__exit__` method or misuse of '__enter__' will result in a runtime error
# AP (Availability + Partition Tolerance): A system that prioritizes availability and partition
# tolerance may use a distributed architecture with eventual consistency (e.g., Cassandra or Riak).
# This ensures that the system is always available (availability), even in the presence of network
# partitions (partition tolerance). However, the system may sacrifice consistency, as nodes may have
# different views of the data (no consistency). A homoiconic piece of source code is eventually
# consistent, assuming it is able to re-instantiated.
def atom(cls: Type[{T, V, C}]) -> Type[{T, V, C}]: # homoicon decorator
    """Decorator to create a homoiconic atom."""
    original_init = cls.__init__
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, 'id'):
            self.id = hashlib.sha256(self.__class__.__name__.encode('utf-8')).hexdigest()
    cls.__init__ = new_init
    return cls
AccessLevel = Enum('READ WRITE EXECUTE ADMIN')
@dataclass
class AccessPolicy:
    level: AccessLevel
    namespace_patterns: list[str] = field(default_factory=list)
    allowed_operations: list[str] = field(default_factory=list)
    def can_access(self, namespace: str, operation: str) -> bool:
        if any(pattern in namespace for pattern in self.namespace_patterns):
            return operation in self.allowed_operations
        return False
class SecurityContext:
    def __init__(self, user_id: str, access_policy: AccessPolicy):
        self.user_id = user_id
        self.access_policy = access_policy
        self._audit_log = []
    def log_access(self, namespace: str, operation: str, success: bool):
        self._audit_log.append({
            "user_id": self.user_id,
            "namespace": namespace,
            "operation": operation,
            "success": success,
            "timestamp": datetime.now().timestamp()
        })
class SecurityValidator(ast.NodeVisitor):
    def __init__(self, security_context: SecurityContext):
        self.security_context = security_context
    def visit_Name(self, node):
        # Validate access to variables
        if not self.security_context.access_policy.can_access(
            node.id, "read"
        ):
            raise PermissionError(f"Access denied to name: {node.id}")
        self.generic_visit(node)
    def visit_Call(self, node):
        # Validate function calls
        if isinstance(node.func, ast.Name):
            if not self.security_context.access_policy.can_access(
                node.func.id, "execute"
            ):
                raise PermissionError(f"Access denied to function: {node.func.id}")
        self.generic_visit(node)
@dataclass
class RuntimeState:
    pdm_installed: bool = False
    virtualenv_created: bool = False
    dependencies_installed: bool = False
    lint_passed: bool = False
    code_formatted: bool = False
    tests_passed: bool = False
    benchmarks_run: bool = False
    pre_commit_installed: bool = False
    variables: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    allowed_root: str = field(init=False)
    def __post_init__(self):
        try:
            self.allowed_root = os.path.dirname(os.path.realpath(__file__))
            if not any(os.listdir(self.allowed_root)):
                raise FileNotFoundError(f"Allowed root directory empty: {self.allowed_root}")
            logging.info(f"Allowed root directory found: {self.allowed_root}")
        except Exception as e:
            logging.error(f"Error initializing FilesystemState: {e}")
            raise
    def safe_remove(self, path: str):
        """Safely remove a file or directory, handling platform-specific issues."""
        try:
            path = os.path.abspath(path)
            if not os.path.commonpath([self.allowed_root, path]) == self.allowed_root:
                logging.error(f"Attempt to delete outside allowed directory: {path}")
                return
            if os.path.isdir(path):
                os.rmdir(path)
                logging.info(f"Removed directory: {path}")
            else:
                os.remove(path)
                logging.info(f"Removed file: {path}")
        except (FileNotFoundError, PermissionError, OSError) as e:
            logging.error(f"Error removing path {path}: {e}")
    @classmethod
    def platform(cls):
        if IS_POSIX:
            from ctypes import cdll
        elif IS_WINDOWS:
            from ctypes import windll
            from ctypes.wintypes import DWORD, HANDLE
        try:
            state = cls()
            cls.ExcludeFilter = ExcludeFilter([])
            tracemalloc.start()
            return state
        except Exception as e:
            logging.warning(f"Failed to initialize runtime state: {e}")
            return None
    async def run_command_async(self, command: str, shell: bool = False, timeout: int = 120):
        logging.info(f"Running command: {command}")
        split_command = shlex.split(command, posix=(os.name == 'posix'))
        try:
            process = await asyncio.create_subprocess_exec(
                *split_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=shell
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            return {
                "return_code": process.returncode,
                "output": stdout.decode() if stdout else "",
                "error": stderr.decode() if stderr else "",
            }
        except asyncio.TimeoutError:
            logging.error(f"Command '{command}' timed out.")
            return {"return_code": -1, "output": "", "error": "Command timed out"}
        except Exception as e:
            logging.error(f"Error running command '{command}': {str(e)}")
            return {"return_code": -1, "output": "", "error": str(e)}
class QueryValidator(ast.NodeVisitor):
    def __init__(self, security_context: SecurityContext):
        self.security_context = security_context
    def visit_Name(self, node):
        # Validate access to variables
        if not self.security_context.access_policy.can_access(
            node.id, "read"
        ):
            raise PermissionError(f"Access denied to name: {node.id}")
        self.generic_visit(node)
    def visit_Call(self, node):
        # Validate function calls
        if isinstance(node.func, ast.Name):
            if not self.security_context.access_policy.can_access(
                node.func.id, "execute"
            ):
                raise PermissionError(f"Access denied to function: {node.func.id}")
        self.generic_visit(node)
def load_modules():
    """Function to load modules into the global runtime manager."""
    manager = RuntimeNamespace()
    manager.root.load_modules()
    return manager.root.available_modules  # Return available modules for access
def reload_module(module):
    try:
        importlib.reload(module)
        return True
    except Exception as e:
        logger.error(f"Error reloading module {module.__name__}: {e}")
        return False
def List_Available_Functions(self): # ADD KWARGS
    return [name for name in dir(self.globals) if callable(getattr(self.globals, name))]
class RuntimeNamespace:
    def __init__(self, name: str, parent: Optional['RuntimeNamespace'] = None):
        self._name = name
        self._parent = parent
        self._children: Dict[str, 'RuntimeNamespace'] = {}
        self._content = SimpleNamespace()
        self._security_context: Optional[SecurityContext] = None
        self.available_modules: Dict[str, ModuleType] = {}  # Store available modules here
    @property
    def full_path(self) -> str:
        if self._parent:
            return f"{self._parent.full_path}.{self._name}"
        return self._name
    def add_child(self, name: str) -> 'RuntimeNamespace':
        child = RuntimeNamespace(name, self)
        self._children[name] = child
        return child
    def get_child(self, path: str) -> Optional['RuntimeNamespace']:
        parts = path.split(".", 1)
        if len(parts) == 1:
            return self._children.get(parts[0])
        child = self._children.get(parts[0])
        if child and len(parts) > 1:
            return child.get_child(parts[1])
        return None
    def load_modules(self):
        """Load available modules into the namespace."""
        try:
            for path in pathlib.Path(__file__).parent.glob("*.py"):
                if path.name.startswith("_"):
                    continue
                module_name = path.stem
                spec = spec_from_file_location(module_name, path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load module {module_name}")
                module = module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                self.available_modules[module_name] = module  # Store in the namespace
            logging.info("Modules loaded successfully.")
        except Exception as e:
            logging.error(f"Error importing internal modules: {e}")
            sys.exit(1)
    def __init__(self):
        self.root = RuntimeNamespace("root")
        self._security_contexts: Dict[str, SecurityContext] = {}
    def register_user(self, user_id: str, access_policy: AccessPolicy):
        self._security_contexts[user_id] = SecurityContext(user_id, access_policy)
    async def execute_query(self, user_id: str, query: str) -> Any:
        security_context = self._security_contexts.get(user_id)
        if not security_context:
            raise PermissionError("User not registered")
        try:
            # Parse query and validate
            parsed = ast.parse(query, mode='eval')
            validator = QueryValidator(security_context)  # Ensure QueryValidator is defined elsewhere
            validator.visit(parsed)
            # Execute in isolated namespace
            namespace = self._create_restricted_namespace(security_context)
            result = eval(compile(parsed, '<string>', 'eval'), namespace)
            security_context.log_access(
                namespace="query_execution",
                operation="execute",
                success=True
            )
            return result
        except Exception as e:
            security_context.log_access(
                namespace="query_execution",
                operation="execute",
                success=False
            )
            logging.error(f"Error executing query: {e}")
            raise
    def _create_restricted_namespace(self, security_context: SecurityContext) -> dict:
        # Create a restricted namespace based on security context
        return {
            "__builtins__": None,  # Disable built-in functions
            "print": print if security_context.access_policy.level >= AccessLevel.READ else None,
            # Add other safe functions as needed
        }
    def isModule(rawClsOrFn: Union[Type, Callable]) -> Optional[str]:
        pyModule = inspect.getmodule(rawClsOrFn)
        if hasattr(pyModule, "__file__"):
            return str(Path(pyModule.__file__).resolve())
        return None
    def getModuleImportInfo(rawClsOrFn: Union[Type, Callable]) -> Tuple[Optional[str], str, str]:
        """
        Given a class or function in Python, get all the information needed to import it in another Python process.
        This version balances portability and optimization using camel case.
        """
        pyModule = inspect.getmodule(rawClsOrFn)
        if pyModule is None or pyModule.__name__ == '__main__':
            return None, 'interactive', rawClsOrFn.__name__
        modulePath = isModule(rawClsOrFn)
        if not modulePath:
            # Built-in or frozen module
            return None, pyModule.__name__, rawClsOrFn.__name__
        rootPath = str(Path(modulePath).parent)
        moduleName = pyModule.__name__
        clsOrFnName = getattr(rawClsOrFn, "__qualname__", rawClsOrFn.__name__)
        if getattr(pyModule, "__package__", None):
            try:
                package = __import__(pyModule.__package__)
                packagePath = str(Path(package.__file__).parent)
                if Path(packagePath) in Path(modulePath).parents:
                    rootPath = str(Path(packagePath).parent)
                else:
                    print(f"Warning: Module is not in the expected package structure. Using file parent as root path.")
            except Exception as e:
                print(f"Warning: Error processing package structure: {e}. Using file parent as root path.")
        return rootPath, moduleName, clsOrFnName
