from __future__ import annotations
import asyncio
import time
import array
import json
import math
import http.client
import os
import ast
import hashlib
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    TypeVar, Generic, Callable, Dict, Any, Optional, Set, Union,
    Awaitable, List, Tuple
)

class __Atom__(ABC):
    __slots__ = ("_refcount",)

    def __init__(self):
        self._refcount = 1

    def inc_ref(self) -> None:
        self._refcount += 1

    def dec_ref(self) -> None:
        self._refcount -= 1
        if self._refcount <= 0:
            self.cleanup()

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources when the atom is no longer referenced."""
        pass
T_co = TypeVar('T_co', covariant=True)
V_co = TypeVar('V_co', covariant=True)
C_co = TypeVar('C_co', bound=Callable, covariant=True)
class AsyncAtom(__Atom__, Generic[T_co, V_co, C_co], ABC):
    """
    An asynchronous implementation of the Atom concept.
    This version is designed for Python 3.13 std libs and integrates with a basic
    __Atom__ interface (including reference counting as per PyObject semantics).
    """
    __slots__ = (
        '_code', '_value', '_local_env', '_ttl', '_created_at',
        '_last_access_time', 'request_data', 'session', 'runtime_namespace',
        'security_context', '_pending_tasks', '_lock', '_buffer_size', '_buffer'
    )

    def __init__(
        self,
        code: str,
        value: Optional[V_co] = None,
        ttl: Optional[int] = None,
        request_data: Optional[Dict[str, Any]] = None,
        buffer_size: int = 1024 * 64  # default 64KB
    ):
        super().__init__()
        self._code = code
        self._value = value
        self._local_env: Dict[str, Any] = {}
        self._ttl = ttl
        self._created_at = time.time()
        self._last_access_time = self._created_at
        self.request_data = request_data or {}
        self.session: Dict[str, Any] = self.request_data.get("session", {})
        self.runtime_namespace = None
        self.security_context = None
        self._pending_tasks: Set[asyncio.Task] = set()
        self._lock = asyncio.Lock()
        self._buffer_size = buffer_size
        self._buffer = bytearray(buffer_size)

    async def __aenter__(self) -> AsyncAtom[T_co, V_co, C_co]:
        self.inc_ref()
        self._last_access_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.dec_ref()
        return False  # Propagate exceptions

    async def cleanup(self) -> None:
        print(f"Cleaning up atom {id(self)}")
        for task in list(self._pending_tasks): # Iterate over a copy as set is modified
            if not task.done():
                task.cancel()
                try:
                    await task # Await cancellation if needed, handle CancelledError
                except asyncio.CancelledError:
                    pass
        self._pending_tasks.clear() # Ensure set is empty
        self._buffer = bytearray(0)
        self._local_env.clear()

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._last_access_time = time.time()
        async with self._lock:
            # Create a controlled environment
            execution_namespace: Dict[str, Any] = {
                '__builtins__': __builtins__, # Or a restricted set
                'asyncio': asyncio,
                'time': time, # Example: expose some modules
                '__atom_self__': self, # Allow access to the atom instance itself
                # Add other necessary imports/objects here
            }
            # Copy local state into the execution namespace
            execution_namespace.update(self._local_env)

        try:
            # Wrap the user code in a standard async function definition
            # This assumes the user code is the *body* of the function
            # A more robust approach would be to require the user code *define* the function
            # For simplicity here, let's assume the code *is* the function body
            # This is still risky if the code isn't just a function body
            # A better way: require the code string *define* `async def atom_entrypoint(...)`
            wrapped_code = f"async def atom_entrypoint(__atom_self__, *args, **kwargs):\n{self._code}"

            code_obj = compile(wrapped_code, '<atom>', 'exec')

            # Execute the definition in the controlled namespace
            exec(code_obj, execution_namespace)

            # Get the defined function
            main_func = execution_namespace.get('atom_entrypoint')

            if not main_func or not asyncio.iscoroutinefunction(main_func):
                raise ValueError("Code must define an async function 'atom_entrypoint'")

            # Call the function, passing necessary context
            result = await main_func(__atom_self__=self, *args, **kwargs)

            async with self._lock:
                # Update shared local environment from the execution namespace
                # Be explicit about what state is shared/persisted
                # Maybe only update keys that were explicitly marked for persistence?
                # Or update all keys *except* builtins, args, kwargs, etc.
                # Let's update all keys except the ones we injected for execution context
                reserved_keys = {'__builtins__', 'asyncio', 'time', '__atom_self__', 'atom_entrypoint', 'args', 'kwargs'} # Add others as needed
                for k, v in execution_namespace.items():
                    if k not in reserved_keys:
                        self._local_env[k] = v # This now allows new variables to persist

            return result
        except Exception as e:
            print(f"Error executing AsyncAtom code: {e}") # Use a proper logger
            raise RuntimeError(f"Error executing AsyncAtom code: {e}") from e # Chain exception

    def _is_async_code(self, code: str) -> bool:
        try:
            parsed = ast.parse(code)
            for node in ast.walk(parsed):
                if isinstance(node, (ast.AsyncFunctionDef, ast.Await)):
                    return True
            return False
        except SyntaxError:
            return False

    async def spawn_task(self, coro: Awaitable) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        return task

    async def handle_request(self, *args: Any, **kwargs: Any) -> Any:
        self._last_access_time = time.time()
        if not await self.is_authenticated():
            return {"status": "error", "message": "Authentication failed"}
        await self.log_request()
        request_context = {
            "session": self.session,
            "request_data": self.request_data,
            "runtime_namespace": self.runtime_namespace,
            "security_context": self.security_context,
            "timestamp": time.time()
        }
        try:
            if "operation" in self.request_data:
                operation = self.request_data["operation"]
                if operation == "execute_atom":
                    result = await self.execute_atom(request_context)
                elif operation == "query_memory":
                    result = await self.query_memory(request_context)
                else:
                    result = {"status": "error",
                              "message": "Unknown operation"}
            else:
                result = await self.process_request(request_context)
        except Exception as e:
            result = {"status": "error", "message": str(e)}
        await self.save_session()
        await self.log_response(result)
        return result

    @abstractmethod
    async def is_authenticated(self) -> bool:
        pass

    @abstractmethod
    async def log_request(self) -> None:
        pass

    @abstractmethod
    async def execute_atom(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def query_memory(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def process_request(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def save_session(self) -> None:
        pass

    @abstractmethod
    async def log_response(self, result: Any) -> None:
        pass

    async def preload_buffer(self, data: bytes) -> None:
        async with self._lock:
            if len(data) <= self._buffer_size:
                self._buffer[:len(data)] = data
            else:
                self._buffer = bytearray(data)
                self._buffer_size = len(data)

    async def get_buffer(self, offset: int = 0, length: Optional[int] = None) -> memoryview:
        async with self._lock:
            if length is None:
                return memoryview(self._buffer)[offset:]
            return memoryview(self._buffer)[offset:offset+length]

    def is_expired(self) -> bool:
        if self._ttl is None:
            return False
        now = time.time()
        return now - self._created_at > self._ttl

    @property
    def code(self) -> str:
        return self._code

    @property
    def value(self) -> Optional[V_co]:
        return self._value

    @property
    def ob_refcnt(self) -> int:
        return self._refcount

    @property
    def ob_ttl(self) -> Optional[int]:
        return self._ttl

    @ob_ttl.setter
    def ob_ttl(self, value: Optional[int]) -> None:
        self._ttl = value


class DemoAtom(AsyncAtom[Any, Any, Any]):
    """A concrete implementation of AsyncAtom for demonstration purposes."""
    
    async def is_authenticated(self) -> bool:
        # Simple demo implementation always returns True
        return True

    async def log_request(self) -> None:
        # Simple demo implementation just prints the request
        print(f"Request logged at {datetime.now()}")

    async def execute_atom(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "success",
            "message": "Atom executed successfully",
            "context": request_context
        }

    async def query_memory(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "success",
            "memory_usage": len(self._buffer),
            "context": request_context
        }

    async def process_request(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "success",
            "message": "Request processed",
            "context": request_context
        }

    async def save_session(self) -> None:
        # Demo implementation - just print session state
        print(f"Session saved: {self.session}")

    async def log_response(self, result: Any) -> None:
        # Demo implementation - just print the response
        print(f"Response logged: {result}")

async def test_main_example():
    async with DemoAtom("    return 'Hello, World!'") as atom:
        result = await atom()
        assert result == 'Hello, World!'
        print("test_main_example passed")

async def test_reference_counting_and_cleanup():
    async with DemoAtom("    return 42") as atom:
        initial_refcount = atom.ob_refcnt
        atom.inc_ref()
        assert atom.ob_refcnt == initial_refcount + 1
        atom.dec_ref()
        assert atom.ob_refcnt == initial_refcount
        await atom.cleanup()
        print("test_reference_counting_and_cleanup passed")

async def test_demoatom_methods():
    async with DemoAtom("    return 'test'") as atom:
        authenticated = await atom.is_authenticated()
        assert authenticated is True
        await atom.log_request()
        result = await atom.execute_atom({})
        assert result["status"] == "success"
        memory = await atom.query_memory({})
        assert memory["status"] == "success"
        processed = await atom.process_request({})
        assert processed["status"] == "success"
        await atom.save_session()
        await atom.log_response({"status": "ok"})
        print("test_demoatom_methods passed")

async def run_tests():
    await test_main_example()
    await test_reference_counting_and_cleanup()
    await test_demoatom_methods()
    print("All tests passed")

if __name__ == "__main__":
    asyncio.run(run_tests())
