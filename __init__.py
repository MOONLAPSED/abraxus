from app.source import target
import pathlib
import socket
import subprocess
import time
import sys
import asyncio
import json
import http.server
import tempfile
import site
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path, PureWindowsPath

import os
import tracemalloc
import logging
import threading
import gc
import weakref
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Set, TypeVar, Union
from contextlib import contextmanager
from functools import wraps
from dataclasses import dataclass
from enum import Enum, auto

# Type variables for generic type hints
T = TypeVar('T')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ObsidianSandbox')

IS_POSIX = os.name == 'posix'
IS_WINDOWS = sys.platform.startswith('win')

# Platform-specific optimizations
if IS_WINDOWS:
    from ctypes import windll
    from ctypes import wintypes
    from ctypes.wintypes import HANDLE, DWORD, LPWSTR, LPVOID, BOOL
    def set_process_priority(priority: int):
        windll.kernel32.SetPriorityClass(wintypes.HANDLE(-1), priority)

elif IS_POSIX:
    import resource

    def set_process_priority(priority: int):
        try:
            os.nice(priority)
        except PermissionError:
            print("Warning: Unable to set process priority. Running with default priority.")

WINDOWS_SANDBOX_DEFAULT_DESKTOP = Path(PureWindowsPath(r'C:\Users\WDAGUtilityAccount\Desktop'))

@dataclass
class SandboxConfig:
    mappings: List['FolderMapping']
    networking: bool = True
    logon_command: str = ""
    virtual_gpu: bool = True

    def to_wsb_config(self) -> Dict:
        """Generate Windows Sandbox configuration"""
        config = {
            'MappedFolders': [mapping.to_wsb_config() for mapping in self.mappings],
            'LogonCommand': {'Command': self.logon_command} if self.logon_command else None,
            'Networking': self.networking,
            'vGPU': self.virtual_gpu
        }
        return config

class SandboxException(Exception):
    """Base exception for sandbox-related errors"""
    pass

class ServerNotResponding(SandboxException):
    """Raised when server is not responding"""
    pass

@dataclass
class FolderMapping:
    """Represents a folder mapping between host and sandbox"""
    host_path: Path
    read_only: bool = True
    
    def __post_init__(self):
        self.host_path = Path(self.host_path)
        if not self.host_path.exists():
            raise ValueError(f"Host path does not exist: {self.host_path}")
    
    @property
    def sandbox_path(self) -> Path:
        """Get the mapped path inside the sandbox"""
        return WINDOWS_SANDBOX_DEFAULT_DESKTOP / self.host_path.name
    
    def to_wsb_config(self) -> Dict:
        """Convert to Windows Sandbox config format"""
        return {
            'HostFolder': str(self.host_path),
            'ReadOnly': self.read_only
        }

class PythonUserSiteMapper:
    def read_only(self):
        return True
    """
    Maps the current Python installation's user site packages to the new sandbox.
    """

    def site(self):
        return pathlib.Path(site.getusersitepackages())

    """
    Maps the current Python installation to the new sandbox.
    """
    def path(self):
        return pathlib.Path(sys.prefix)

class OnlineSession:
    """Manages the network connection to the sandbox"""
    def __init__(self, sandbox: 'SandboxEnvironment'):
        self.sandbox = sandbox
        self.shared_directory = self._get_shared_directory()
        self.server_address_path = self.shared_directory / 'server_address'
        self.server_address_path_in_sandbox = self._get_sandbox_server_path()

    def _get_shared_directory(self) -> Path:
        """Create and return shared directory path"""
        shared_dir = Path(tempfile.gettempdir()) / 'obsidian_sandbox_shared'
        shared_dir.mkdir(exist_ok=True)
        return shared_dir

    def _get_sandbox_server_path(self) -> Path:
        """Get the server address path as it appears in the sandbox"""
        return WINDOWS_SANDBOX_DEFAULT_DESKTOP / self.shared_directory.name / 'server_address'

    def configure_sandbox(self):
        """Configure sandbox for network communication"""
        self.sandbox.config.mappings.append(
            FolderMapping(self.shared_directory, read_only=False)
        )
        self._setup_logon_script()

    def _setup_logon_script(self):
        """Generate logon script for sandbox initialization"""
        commands = []
        
        # Setup Python environment
        python_path = sys.executable
        sandbox_python_path = WINDOWS_SANDBOX_DEFAULT_DESKTOP / 'Python' / 'python.exe'
        commands.append(f'copy "{python_path}" "{sandbox_python_path}"')
        
        # Start server
        commands.append(f'{sandbox_python_path} -m http.server 8000')
        
        self.sandbox.config.logon_command = 'cmd.exe /c "{}"'.format(' && '.join(commands))

    def connect(self, timeout: int = 60) -> Tuple[str, int]:
        """Establish connection to sandbox"""
        if self._wait_for_file(timeout):
            address, port = self.server_address_path.read_text().strip().split(':')
            if self._verify_connection(address, int(port)):
                return address, int(port)
            raise ServerNotResponding("Server is not responding")
        raise SandboxException("Failed to establish connection")

    def _wait_for_file(self, timeout: int) -> bool:
        """Wait for server address file creation"""
        end_time = time.time() + timeout
        while time.time() < end_time:
            if self.server_address_path.exists():
                return True
            time.sleep(1)
        return False

    def _verify_connection(self, address: str, port: int) -> bool:
        """Verify network connection to sandbox"""
        try:
            with socket.create_connection((address, port), timeout=3):
                return True
        except (socket.error, socket.timeout):
            return False

class SandboxEnvironment:
    """Manages the Windows Sandbox environment"""
    def __init__(self, config: SandboxConfig):
        self.config = config
        self._session = OnlineSession(self)
        self._connection: Optional[Tuple[str, int]] = None
        
        if config.networking:
            self._session.configure_sandbox()
            self._connection = self._session.connect()

    def run_executable(self, executable_args: List[str], **kwargs) -> subprocess.Popen:
        """Run an executable in the sandbox"""
        kwargs.setdefault('stdout', subprocess.PIPE)
        kwargs.setdefault('stderr', subprocess.PIPE)
        return subprocess.Popen(executable_args, **kwargs)

    def shutdown(self):
        """Safely shutdown the sandbox"""
        try:
            self.run_executable(['shutdown.exe', '/s', '/t', '0'])
        except Exception as e:
            logger.error(f"Failed to shutdown sandbox: {e}")
            raise SandboxException("Shutdown failed")

class SandboxCommServer:
    """Manages communication with the sandbox environment"""
    def __init__(self, shared_dir: Path):
        self.shared_dir = shared_dir
        self.server: Optional[http.server.HTTPServer] = None
        self._port = self._find_free_port()
    
    @staticmethod
    def _find_free_port() -> int:
        """Find an available port for the server"""
        with socket.socket() as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    async def start(self):
        """Start the communication server"""
        class Handler(http.server.SimpleHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers['Content-Length'])
                data = self.rfile.read(content_length)
                # Process incoming messages from sandbox
                logger.info(f"Received from sandbox: {data.decode()}")
                self.send_response(200)
                self.end_headers()
        
        self.server = http.server.HTTPServer(('localhost', self._port), Handler)
        
        # Write server info for sandbox
        server_info = {'host': 'localhost', 'port': self._port}
        server_info_path = self.shared_dir / 'server_info.json'
        server_info_path.write_text(json.dumps(server_info))
        
        # Run server in background
        await asyncio.get_event_loop().run_in_executor(
            None, self.server.serve_forever
        )
    
    def stop(self):
        """Stop the communication server"""
        if self.server:
            self.server.shutdown()
            self.server = None

class SandboxManager:
    """Manages Windows Sandbox lifecycle and communication"""
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.shared_dir = Path(tempfile.gettempdir()) / 'sandbox_shared'
        self.shared_dir.mkdir(exist_ok=True)
        
        # Add shared directory to mappings
        self.config.mappings.append(
            FolderMapping(self.shared_dir, read_only=False)
        )
        
        self.comm_server = SandboxCommServer(self.shared_dir)
        self._process: Optional[subprocess.Popen] = None
    
    async def _setup_sandbox(self):
        """Generate WSB file and prepare sandbox environment"""
        wsb_config = self.config.to_wsb_config()
        wsb_path = self.shared_dir / 'config.wsb'
        wsb_path.write_text(json.dumps(wsb_config, indent=2))
        
        # Start communication server
        await self.comm_server.start()
        
        # Launch sandbox
        self._process = subprocess.Popen(
            ['WindowsSandbox.exe', str(wsb_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    async def _cleanup(self):
        """Clean up sandbox resources"""
        self.comm_server.stop()
        if self._process:
            self._process.terminate()
            await asyncio.get_event_loop().run_in_executor(
                None, self._process.wait
            )

    @asynccontextmanager
    async def session(self) -> AsyncIterator['SandboxManager']:
        """Context manager for sandbox session"""
        try:
            await self._setup_sandbox()
            yield self
        finally:
            await self._cleanup()


class MemoryTraceLevel(Enum):
    """Granularity levels for memory tracing."""
    BASIC = auto()      # Basic memory usage
    DETAILED = auto()   # Include stack traces
    FULL = auto()       # Include object references

@dataclass
class MemoryStats:
    """Container for memory statistics with analysis capabilities."""
    size: int
    count: int
    traceback: str
    timestamp: float
    peak_memory: int
    
    def to_dict(self) -> Dict:
        return {
            'size': self.size,
            'count': self.count,
            'traceback': self.traceback,
            'timestamp': self.timestamp,
            'peak_memory': self.peak_memory
        }

class CustomFormatter(logging.Formatter):
    """Custom formatter for color-coded log levels."""
    COLORS = {
        logging.DEBUG: "\x1b[38;20m",
        logging.INFO: "\x1b[32;20m",
        logging.WARNING: "\x1b[33;20m",
        logging.ERROR: "\x1b[31;20m",
        logging.CRITICAL: "\x1b[31;1m"
    }
    RESET = "\x1b[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.COLORS[logging.DEBUG])
        record.msg = f"{color}{record.msg}{self.RESET}"
        return super().format(record)

class MemoryTracker:
    """Singleton memory tracking manager with enhanced logging."""
    _instance = None
    _lock = threading.Lock()
    _trace_filter = {"<frozen importlib._bootstrap>", "<frozen importlib._bootstrap_external>"}
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """Initialize the memory tracker with logging and storage."""
        self._setup_logging()
        self._snapshots: Dict[str, List[MemoryStats]] = {}
        self._tracked_objects = weakref.WeakSet()
        self._trace_level = MemoryTraceLevel.DETAILED
        
        # Start tracemalloc if not already running
        if not tracemalloc.is_tracing():
            tracemalloc.start()
    
    def _setup_logging(self):
        """Configure logging with custom formatter."""
        self.logger = logging.getLogger("MemoryTracker")
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler with color formatting
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(CustomFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        self.logger.addHandler(console_handler)
        
        # File handler for persistent logging
        try:
            file_handler = logging.FileHandler("memory_tracker.log")
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            self.logger.addHandler(file_handler)
        except (PermissionError, IOError) as e:
            self.logger.warning(f"Could not create log file: {e}")

def trace_memory(level: MemoryTraceLevel = MemoryTraceLevel.DETAILED):
    """Enhanced decorator for memory tracking with configurable detail level."""
    def decorator(method: Callable) -> Callable:
        @wraps(method)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            tracker = MemoryTracker()
            
            # Force garbage collection for accurate measurement
            gc.collect()
            
            # Take initial snapshot
            snapshot_before = tracemalloc.take_snapshot()
            
            try:
                result = method(self, *args, **kwargs)
                
                # Take final snapshot and compute statistics
                snapshot_after = tracemalloc.take_snapshot()
                stats = snapshot_after.compare_to(snapshot_before, 'lineno')
                
                # Filter and process statistics
                filtered_stats = [
                    stat for stat in stats 
                    if not any(f in str(stat.traceback) for f in tracker._trace_filter)
                ]
                
                # Log based on trace level
                if level in (MemoryTraceLevel.DETAILED, MemoryTraceLevel.FULL):
                    for stat in filtered_stats[:5]:
                        tracker.logger.info(
                            f"Memory change in {method.__name__}: "
                            f"+{stat.size_diff/1024:.1f} KB at:\n{stat.traceback}"
                        )
                
                return result
                
            finally:
                # Cleanup
                del snapshot_before
                gc.collect()
                
        return wrapper
    return decorator

class MemoryTrackedABC(ABC):
    """Abstract base class for memory-tracked classes with enhanced features."""
    
    def __init__(self):
        self._tracker = MemoryTracker()
        self._tracker._tracked_objects.add(self)
    
    def __init_subclass__(cls):
        super().__init_subclass__()
        
        # Store original methods for introspection
        cls._original_methods = {}
        
        # Automatically decorate public methods
        for attr_name, attr_value in cls.__dict__.items():
            if (callable(attr_value) and 
                not attr_name.startswith('_') and 
                not getattr(attr_value, '_skip_trace', False)):
                cls._original_methods[attr_name] = attr_value
                setattr(cls, attr_name, trace_memory()(attr_value))
    
    @staticmethod
    def skip_trace(method: Callable) -> Callable:
        """Decorator to exclude a method from memory tracking."""
        method._skip_trace = True
        return method
    
    @classmethod
    @contextmanager
    def trace_section(cls, section_name: str, level: MemoryTraceLevel = MemoryTraceLevel.DETAILED):
        """Context manager for tracking memory usage in specific code sections."""
        tracker = MemoryTracker()
        
        gc.collect()
        snapshot_before = tracemalloc.take_snapshot()
        
        try:
            yield
        finally:
            snapshot_after = tracemalloc.take_snapshot()
            stats = snapshot_after.compare_to(snapshot_before, 'lineno')
            
            filtered_stats = [
                stat for stat in stats 
                if not any(f in str(stat.traceback) for f in tracker._trace_filter)
            ]
            
            if level != MemoryTraceLevel.BASIC:
                tracker.logger.info(f"\nMemory usage for section '{section_name}':")
                for stat in filtered_stats[:5]:
                    tracker.logger.info(f"{stat}")
            
            del snapshot_before
            gc.collect()








class DebuggerMixin:
    """Mixin for debugging memory-tracked classes."""

    def __init__(self):
        self._tracker = MemoryTracker()
        self._tracker._tracked_objects.add(self)

    def __init_subclass__(cls):
        super().__init_subclass__()

        # Store original methods for introspection
        cls._original_methods = {}

        # Automatically decorate public methods
        for attr_name, attr_value in cls.__dict__.items():
            if (callable(attr_value) and
                not attr_name.startswith('_') and
                not getattr(attr_value, '_skip_trace', False)):
                cls._original_methods[attr_name] = attr_value
                setattr(cls, attr_name, trace_memory()(attr_value))

    @staticmethod
    def skip_trace(method: Callable) -> Callable:
        """Decorator to exclude a method from memory tracking."""
        method._skip_trace = True
        return method

    @classmethod
    @contextmanager
    def trace_section(cls, section_name: str, level: MemoryTraceLevel = MemoryTraceLevel.DETAILED):
        """Context manager for tracking memory usage in specific code sections."""
        tracker = MemoryTracker()

def main():
    class MyTrackedClass(MemoryTrackedABC):
        def tracked_method(self):
            """This method will be automatically tracked with detailed memory info."""
            large_list = [i for i in range(1000000)]
            return sum(large_list)
        
        @MemoryTrackedABC.skip_trace
        def untracked_method(self):
            """This method will not be tracked."""
            return "Not tracked"
        
        def tracked_with_section(self):
            """Example of using trace_section with different detail levels."""
            with self.trace_section("initialization", MemoryTraceLevel.BASIC):
                result = []
                
            with self.trace_section("processing", MemoryTraceLevel.DETAILED):
                result.extend(i * 2 for i in range(500000))
                
            with self.trace_section("cleanup", MemoryTraceLevel.FULL):
                result.clear()
                
            return len(result)
    
        @classmethod
        def introspect_methods(cls):
            """Introspect and display tracked methods with their original implementations."""
            for method_name, original_method in cls._original_methods.items():
                print(f"Method: {method_name}")
                print(f"Original implementation: {original_method}")
                print("---")

            return MyTrackedClass()
    return MyTrackedClass()

if __name__ == "__main__":
    tracker = MemoryTracker()
    tracker.logger.setLevel(logging.DEBUG)
    tracker.logger.addHandler(logging.StreamHandler())
    tracker.logger.addHandler(logging.FileHandler("memory_tracker.log"))
    my_instance = main()
    my_instance.__class__.introspect_methods()

    MyTrackedClass = main().__class__
    MyTrackedClass.introspect_methods()


    # Basic usage
    obj = MyTrackedClass()
    obj.tracked_method()  # Automatically tracked with detailed info

    # Custom section tracking with different detail levels
    obj.tracked_with_section()

    # Customize tracking level for specific methods
    @trace_memory(level=MemoryTraceLevel.FULL)
    def custom_tracked_method(self):
        pass