import ast
import hashlib
import importlib.util
import json
import logging
import mimetypes
import os
import pathlib
from abc import ABC, abstractmethod
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from queue import Queue
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union, cast, Generic
from contextlib import contextmanager
T = TypeVar('T')
V = TypeVar('V')
C = TypeVar('C')
@dataclass
class AppConfig:
    root_dir: pathlib.Path = pathlib.Path.cwd()
    log_level: int = logging.INFO
    allowed_extensions: Set[str] = field(default_factory=lambda: {'.py', '.txt', '.md'})
    admin_users: Set[str] = field(default_factory=lambda: {'admin'})
    db_connection_string: Optional[str] = None
    cache_ttl: int = 3600  # seconds
    max_threads: int = 10
    request_timeout: int = 30  # seconds
    max_retry_attempts: int = 3
    max_file_size: int = 10485760  # bytes (10 MB)
    retry_backoff_factor: float = 1.5
    environment: str = "development"
    thread_pool_size: int = field(default_factory=lambda: int(os.environ.get('APP_THREAD_POOL_SIZE', '4')))
    enable_security: bool = field(default_factory=lambda: os.environ.get('APP_ENABLE_SECURITY', 'true').lower() == 'true')
    temp_dir: pathlib.Path = field(default_factory=lambda: pathlib.Path(os.environ.get('APP_TEMP_DIR', '/tmp')))
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create configuration from environment variables."""
        return cls(
            root_dir=pathlib.Path(os.getenv('APP_ROOT_DIR', str(pathlib.Path.cwd()))),
            log_level=getattr(logging, os.getenv('APP_LOG_LEVEL', 'INFO')),
            allowed_extensions=set(os.getenv('APP_ALLOWED_EXTENSIONS', '.py,.txt,.md').split(',')),
            admin_users=set(os.getenv('APP_ADMIN_USERS', 'admin').split(',')),
            db_connection_string=os.getenv('APP_DB_CONNECTION_STRING'),
            cache_ttl=int(os.getenv('APP_CACHE_TTL', '3600')),
            max_threads=int(os.getenv('APP_MAX_THREADS', '10')),
            request_timeout=int(os.getenv('APP_REQUEST_TIMEOUT', '30')),
            max_retry_attempts=int(os.getenv('APP_MAX_RETRY_ATTEMPTS', '3')),
            retry_backoff_factor=float(os.getenv('APP_RETRY_BACKOFF_FACTOR', '1.5')),
            environment=os.getenv('APP_ENVIRONMENT', 'development')
        )
    @classmethod
    def from_file(cls, config_path: Union[str, pathlib.Path]) -> 'AppConfig':
        """Load configuration from a JSON file."""
        path = pathlib.Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, 'r') as f:
            config_dict = json.load(f)
        # Convert string paths to Path objects
        if 'root_dir' in config_dict:
            config_dict['root_dir'] = pathlib.Path(config_dict['root_dir'])
        if 'temp_dir' in config_dict:
            config_dict['temp_dir'] = pathlib.Path(config_dict['temp_dir'])
        return cls(**config_dict)
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, handling Path objects."""
        result = asdict(self)
        result['root_dir'] = str(self.root_dir)
        result['temp_dir'] = str(self.temp_dir)
        return result
    def save(self, path: Union[str, pathlib.Path]) -> None:
        """Save configuration to a JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
class PerformanceMetrics:
    """Track and report performance metrics."""
    def __init__(self):
        self._metrics = {}
        self._lock = threading.RLock()
    @contextmanager
    def measure(self, operation_name: str):
        """Context manager to measure operation duration."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            with self._lock:
                if operation_name not in self._metrics:
                    self._metrics[operation_name] = {'count': 0, 'total_time': 0, 'min_time': float('inf'), 'max_time': 0}
                self._metrics[operation_name]['count'] += 1
                self._metrics[operation_name]['total_time'] += duration
                self._metrics[operation_name]['min_time'] = min(self._metrics[operation_name]['min_time'], duration)
                self._metrics[operation_name]['max_time'] = max(self._metrics[operation_name]['max_time'], duration)
    def get_report(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics report with averages."""
        with self._lock:
            report = {}
            for op_name, stats in self._metrics.items():
                avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
                report[op_name] = {
                    'count': stats['count'],
                    'avg_time': avg_time,
                    'min_time': stats['min_time'] if stats['min_time'] != float('inf') else 0,
                    'max_time': stats['max_time']
                }
            return report
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._metrics = {}
# Global performance metrics instance
performance_metrics = PerformanceMetrics()
def measure_performance(func):
    """Decorator to measure function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with performance_metrics.measure(func.__qualname__):
            return func(*args, **kwargs)
    return wrapper
class ThreadSafeFormatter(logging.Formatter):
    """Thread-safe formatter with color support."""
    COLORS = {
        logging.DEBUG: "\x1b[36m",     # Cyan
        logging.INFO: "\x1b[32m",      # Green
        logging.WARNING: "\x1b[33m",   # Yellow
        logging.ERROR: "\x1b[31m",     # Red
        logging.CRITICAL: "\x1b[31;1m" # Bold red
    }
    RESET = "\x1b[0m"
    FORMAT = "%(asctime)s - [%(threadName)s] - %(name)s - %(levelname)s - %(message)s"
    JSON_FORMAT = False
    def format(self, record):
        # Add correlation ID if available
        if hasattr(threading.current_thread(), 'correlation_id'):
            record.correlation_id = threading.current_thread().correlation_id
        if hasattr(record, 'structured') and record.structured and self.JSON_FORMAT:
            # Create structured log entry
            log_data = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'thread': record.thread,
                'thread_name': record.threadName
            }
            # Add extra attributes
            for key, value in record.__dict__.items():
                if key not in log_data and not key.startswith('_') and isinstance(value, (str, int, float, bool, type(None))):
                    log_data[key] = value
            return json.dumps(log_data)
        else:
            # Standard colored log format
            message = super().format(record)
            color = self.COLORS.get(record.levelno, "")
            return f"{color}{message}{self.RESET}" if color else message
class AppLogger(logging.Logger):
    """Enhanced logger with structured logging support."""
    def structured(self, level, msg, *args, **kwargs):
        """Log with structured data for easier parsing."""
        extra = kwargs.pop('extra', {}) or {}
        extra['structured'] = True
        kwargs['extra'] = extra
        self.log(level, msg, *args, **kwargs)
class LogFilter(logging.Filter):
    """Filter that can exclude specific log patterns."""
    def __init__(self, exclude_patterns: List[str] = None):
        super().__init__()
        self.exclude_patterns = exclude_patterns or []
    def filter(self, record):
        message = record.getMessage()
        return not any(pattern in message for pattern in self.exclude_patterns)
def setup_logging(config: AppConfig) -> logging.Logger:
    """Configure application logging with file and console handlers."""
    # Register custom logger class
    logging.setLoggerClass(AppLogger)
    logger = logging.getLogger("app")
    logger.setLevel(config.log_level)
    if logger.hasHandlers():
        logger.handlers.clear()  # Clear existing handlers
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(config.log_level)
    ch.setFormatter(ThreadSafeFormatter())
    if config.environment != "development":
        logs_dir = config.root_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        fh = logging.FileHandler(logs_dir / f"app-{datetime.now().strftime('%Y%m%d')}.log")
        fh.setLevel(logging.ERROR)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(correlation_id)s - %(message)s'
        ))
        logger.addHandler(fh)
    else:  # non-dev logging (json-formatted)
        log_file = config.root_dir / "app.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(config.log_level)
        file_formatter = ThreadSafeFormatter()
        file_formatter.JSON_FORMAT = True  # Use JSON format for file logs
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    # Add filter to exclude noisy logs
    exclude_filter = LogFilter(exclude_patterns=["heartbeat", "routine check", "frozen importlib._bootstrap", "frozen importlib._bootstrap_external"])
    ch.addFilter(exclude_filter)
    logger.addHandler(ch)
    return logger
class AppError(Exception):
    """Base exception for application errors."""
    def __init__(self, message: str, error_code: str = "APP_ERROR", status_code: int = 420):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.timestamp = datetime.now()
        super().__init__(message)  # 420
class ConfigError(AppError):
    """Configuration related errors."""
    def __init__(self, message: str):
        super().__init__(message, "CONFIG_ERROR")  # 500
class SecurityError(AppError):
    """Security related errors."""
    def __init__(self, message: str):
        super().__init__(message, "SECURITY_ERROR")  # 403
class ContentError(AppError):
    """Content related errors."""
    def __init__(self, message: str):
        super().__init__(message, "CONTENT_ERROR")  # 400
class NamespaceError(AppError):
    """Namespace related errors."""
    def __init__(self, message: str):
        super().__init__(message, "NAMESPACE_ERROR") # 404
def error_handler(logger):
    """Decorator for standardized error handling."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AppError as e:
                logger.error(f"{e.__class__.__name__}: {e.message}", 
                             exc_info=True, 
                             extra={'status_code': e.status_code})
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}", exc_info=True)
                raise AppError(f"An unexpected error occurred: {str(e)}")
        return wrapper
    return decorator
def retry(max_attempts: int = 3, backoff_factor: float = 1.5, 
          exceptions: tuple = (Exception,), logger: Optional[logging.Logger] = None):
    """Decorator to retry functions with exponential backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            attempt = 1
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    wait_time = backoff_factor ** (attempt - 1)
                    if logger:
                        logger.warning(
                            f"Attempt {attempt}/{max_attempts} for {func.__name__} failed: {e}. "
                            f"Retrying in {wait_time:.2f}s."
                        )
                    if attempt == max_attempts:
                        raise
                    time.sleep(wait_time)
                    attempt += 1
        return wrapper
    return decorator
class AccessLevel(Enum):
    READ = 1
    WRITE = 2
    EXECUTE = 3
    ADMIN = 4
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

@dataclass
class AccessPolicy:
    """Defines access control policies with pattern matching."""
    level: AccessLevel
    namespace_patterns: List[str] = field(default_factory=list)
    allowed_operations: Set[str] = field(default_factory=set)
    expiration: Optional[datetime] = None
    @measure_performance
    def can_access(self, namespace: str, operation: str) -> bool:
        """Check if operation is allowed for the given namespace."""
        # Check expiration
        if self.expiration and datetime.now() > self.expiration:
            return False
        # Check namespace pattern match and operation
        matches_pattern = any(
            self._match_pattern(p, namespace) for p in self.namespace_patterns
        )
        return matches_pattern and operation in self.allowed_operations
    def _match_pattern(self, pattern: str, namespace: str) -> bool:
        """Match namespace against pattern with wildcards."""
        if pattern == "*":  # Wildcard matches everything
            return True
        # Handle patterns with wildcards (simple glob-like implementation)
        if "*" in pattern:
            parts = pattern.split("*")
            if not namespace.startswith(parts[0]):
                return False
            current_pos = len(parts[0])
            for part in parts[1:]:
                if not part:  # Empty part (e.g., trailing asterisk)
                    continue
                pos = namespace.find(part, current_pos)
                if pos == -1:
                    return False
                current_pos = pos + len(part)
            # If last part and no trailing asterisk, must end with that part
            if parts[-1] and not pattern.endswith("*"):
                return namespace.endswith(parts[-1])
            return True
        # Exact match
        return namespace == pattern
class SecurityContext:
    """Manages security context with audit trail."""
    def __init__(self, user_id: str, access_policy: AccessPolicy, logger=None):
        self.user_id = user_id
        self.access_policy = access_policy
        self.logger = logger
        self._audit_log: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._last_audit_flush = datetime.now()
        self._audit_flush_interval = 60  # seconds

    @measure_performance
    def check_access(self, namespace: str, operation: str) -> bool:
        """Check if user can access the namespace with given operation."""
        result = self.access_policy.can_access(namespace, operation)
        self.log_access(namespace, operation, result)
        return result

    def enforce_access(self, namespace: str, operation: str) -> None:
        """Enforce access control, raising exception if access denied."""
        if not self.check_access(namespace, operation):
            raise SecurityError(
                f"Access denied: User '{self.user_id}' cannot '{operation}' on '{namespace}'"
            )
    def log_access(self, namespace: str, operation: str, success: bool):
        """Log access attempt to audit trail."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user": self.user_id,
            "namespace": namespace,
            "operation": operation,
            "success": success
        }
        with self._lock:
            self._audit_log.append(entry)
            # Log to application logger if available
            if self.logger:
                level = logging.INFO if success else logging.WARNING
                self.logger.structured(
                    level,
                    f"Access {'allowed' if success else 'denied'}: {self.user_id} -> {operation} on {namespace}",
                    extra={
                        'audit': True,
                        'user_id': self.user_id,
                        'namespace': namespace,
                        'operation': operation,
                        'success': success
                    }
                )
            # Periodically flush audit log if it's getting large
            if (len(self._audit_log) > 1000 or 
                (datetime.now() - self._last_audit_flush).total_seconds() > self._audit_flush_interval):
                self._flush_audit_log()
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get copy of the audit log."""
        with self._lock:
            return self._audit_log.copy()
    def _flush_audit_log(self):
        """Flush audit log to persistent storage."""
        # In production, write to database or specialized audit log system
        # For now, just trim the in-memory log
        with self._lock:
            if len(self._audit_log) > 1000:
                self._audit_log = self._audit_log[-1000:]
            self._last_audit_flush = datetime.now()

@dataclass
class FileMetadata:
    path: pathlib.Path
    mime_type: str
    size: int
    created: datetime
    modified: datetime
    content_hash: str
    symlinks: List[pathlib.Path] = field(default_factory=list)
    encoding: str = 'utf-8'
    tags: Set[str] = field(default_factory=set)
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        result = asdict(self)
        result['path'] = str(self.path)
        result['symlinks'] = [str(path) for path in self.symlinks]
        result['created'] = self.created.isoformat()
        result['modified'] = self.modified.isoformat()
        return result
class ContentChangeEvent(Enum):
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
class ContentObserver:
    """Interface for content change observers."""
    def notify(self, event: ContentChangeEvent, metadata: FileMetadata) -> None:
        """Handle content change notification."""
        pass
"""
class ContentManager:
    def __init__(self, config: AppConfig, logger: Optional[logging.Logger] = None):
        self.root_dir = config.root_dir
        self.allowed_extensions = config.allowed_extensions
        self.logger = logger
        self.metadata_cache: Dict[pathlib.Path, FileMetadata] = {}
        self.module_cache: Dict[str, Any] = {}
        self.observers: List[ContentObserver] = []
        self._content_lock = threading.RLock()
        self._observer_lock = threading.RLock()
        self._last_scan_time = datetime.min
    def register_observer(self, observer: ContentObserver) -> None:
        with self._observer_lock:
            self.observers.append(observer)
    def unregister_observer(self, observer: ContentObserver) -> None:
        with self._observer_lock:
            self.observers.remove(observer)
    def notify_observers(self, event: ContentChangeEvent, metadata: FileMetadata) -> None:
        with self._observer_lock:
            for observer in self.observers:
                try:
                    observer.notify(event, metadata)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Observer notification failed: {e}")
    @retry(max_attempts=3, backoff_factor=1.5, exceptions=(IOError,))
    def compute_hash(self, path: pathlib.Path) -> str:
        hasher = hashlib.sha256()
        try:
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except IOError as e:
            if self.logger:
                self.logger.error(f"Failed to compute hash for {path}: {e}")
            raise ContentError(f"Failed to compute hash for {path}: {e}")
    def get_metadata(self, path: pathlib.Path, force_refresh: bool = False) -> FileMetadata:
        with self._content_lock:
            # Return cached metadata if available and not forced refresh
            if not force_refresh and path in self.metadata_cache:
                return self.metadata_cache[path]
            try:
                if not path.exists():
                    raise ContentError(f"File not found: {path}")
                stat = path.stat()
                mime_type, _ = mimetypes.guess_type(path)
                symlinks = [
                    p for p in path.parent.glob('*')
                    if p.is_symlink() and p.resolve() == path
                ]
                # Create new metadata
                metadata = FileMetadata(
                    path=path,
                    mime_type=mime_type or 'application/octet-stream',
                    size=stat.st_size,
                    created=datetime.fromtimestamp(stat.st_ctime),
                    modified=datetime.fromtimestamp(stat.st_mtime),
                    content_hash=self.compute_hash(path),
                    symlinks=symlinks,
                    version=1
                )
                # Update version if previously existed
                if path in self.metadata_cache:
                    old_metadata = self.metadata_cache[path]
                    if old_metadata.content_hash != metadata.content_hash:
                        metadata.version = old_metadata.version + 1
                        self.notify_observers(ContentChangeEvent.MODIFIED, metadata)
                
                self.metadata_cache[path] = metadata
                return metadata
            except Exception as e:
                error_msg = f"Failed to get metadata for {path}: {e}"
                if self.logger:
                    self.logger.error(error_msg)
                raise ContentError(error_msg)
    def load_module(self, path: pathlib.Path, force_reload: bool = False) -> Optional[Any]:
        if path.suffix not in self.allowed_extensions:
            if self.logger:
                self.logger.debug(f"Skipping {path}: extension not allowed")
            return None
        module_name = f"content_{path.stem}"
        with self._content_lock:
            # Return cached module if available and not forced reload
            if not force_reload and module_name in self.module_cache:
                return self.module_cache[module_name]
            try:
                spec = importlib.util.spec_from_file_location(module_name, str(path))
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    metadata = self.get_metadata(path)
                    setattr(module, '__metadata__', metadata)
                    # Load content based on file type
                    if path.suffix in {'.txt', '.md'}:
                        setattr(module, '__content__', path.read_text())
                    elif path.suffix == '.py':
                        spec.loader.exec_module(module)
                        setattr(module, '__content__', path.read_text())
                    self.module_cache[module_name] = module
                    # Notify observers about new content
                    if force_reload:
                        self.notify_observers(ContentChangeEvent.MODIFIED, metadata)
                    else:
                        self.notify_observers(ContentChangeEvent.CREATED, metadata)
                    return module
                else:
                    if self.logger:
                        self.logger.warning(f"Could not create spec for {path}")
                    return None
            except Exception as e:
                error_msg = f"Failed to load module {path}: {e}"
                if self.logger:
                    self.logger.error(error_msg)
                return None
    def scan_directory(self, incremental: bool = True) -> Dict[str, int]:
        result = {"loaded": 0, "modified": 0, "failed": 0}
        scan_time = datetime.now()
        try:
            for path in self.root_dir.rglob('*'):
                if not path.is_file() or path.suffix not in self.allowed_extensions:
                    continue
                try:
                    # Skip unmodified files in incremental scan
                    if incremental and path in self.metadata_cache:
                        metadata = self.metadata_cache[path]
                        if metadata.modified >= self._last_scan_time:
                            # File was modified since last scan
                            self.load_module(path, force_reload=True)
                            result["modified"] += 1
                        continue
                    # Load new or modified file
                    if self.load_module(path):
                        result["loaded"] += 1
                    else:
                        result["failed"] += 1
                except Exception as e:
                    result["failed"] += 1
                    if self.logger:
                        self.logger.error(f"Error processing {path}: {e}")
            # Clean up deleted files
            if incremental:
                cached_paths = list(self.metadata_cache.keys())
                for path in cached_paths:
                    if not path.exists():
                        with self._content_lock:
                            if path in self.metadata_cache:
                                metadata = self.metadata_cache[path]
                                del self.metadata_cache[path]
                                module_name = f"content_{path.stem}"
                                if module_name in self.module_cache:
                                    del self.module_cache[module_name]
                                self.notify_observers(ContentChangeEvent.DELETED, metadata)
            self._last_scan_time = scan_time
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Directory scan failed: {e}")
            raise ContentError(f"Directory scan failed: {e}")
"""
class ContentCache:
    """LRU cache for file content with TTL support."""  
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl  # Time-to-live in seconds
        self._cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, timestamp)
        self._lock = threading.RLock()
    def __len__(self) -> int:
        """Return the number of items in the cache."""
        with self._lock:
            return len(self._cache)
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if exists and not expired."""
        with self._lock:
            if key not in self._cache:
                return None
            value, timestamp = self._cache[key]
            if time.time() - timestamp > self.ttl:
                # Expired entry
                del self._cache[key]
                return None
            # Move to end (most recently used)
            self._cache[key] = (value, timestamp)
            return value
    def set(self, key: str, value: Any) -> None:
        """Add or update value in cache."""
        with self._lock:
            # Evict oldest entry if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            # Add with current timestamp
            self._cache[key] = (value, time.time())
    def invalidate(self, key: str) -> None:
        """Remove entry from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()
class ContentManager:
    """Manages file content with caching and metadata tracking."""
    def __init__(self, config: AppConfig, logger=None):
        self.root_dir = config.root_dir
        self.allowed_extensions = config.allowed_extensions
        self.max_file_size = config.max_file_size
        self.logger = logger
        self.metadata_cache = ContentCache(max_size=1000, ttl=config.cache_ttl)
        self.content_cache = ContentCache(max_size=100, ttl=config.cache_ttl)
        self.module_cache = ContentCache(max_size=50, ttl=config.cache_ttl)
        self._file_watchers: Dict[pathlib.Path, float] = {}  # path -> last_check_time
        self._lock = threading.RLock()
        self._watcher_thread = None
        self._stop_event = threading.Event()
    @measure_performance
    def compute_hash(self, path: pathlib.Path) -> str:
        """Compute SHA256 hash of file content."""
        hasher = hashlib.sha256()
        try:
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except IOError as e:
            if self.logger:
                self.logger.error(f"Failed to compute hash for {path}: {e}")
            raise ContentError(f"Failed to compute hash for {path}: {e}")
    @measure_performance
    def get_metadata(self, path: pathlib.Path) -> FileMetadata:
        """Get or create file metadata with caching."""
        cache_key = str(path.absolute())
        # Check cache first
        cached = self.metadata_cache.get(cache_key)
        if cached:
            return cached
        with self._lock:
            try:
                if not path.exists():
                    raise ContentError(f"File not found: {path}")
                if not path.is_file():
                    raise ContentError(f"Not a file: {path}")
                stat = path.stat()
                # Check file size
                if stat.st_size > self.max_file_size:
                    raise ContentError(f"File exceeds maximum size ({self.max_file_size} bytes): {path}")
                mime_type, _ = mimetypes.guess_type(str(path))
                symlinks = [
                    p for p in path.parent.glob('*')
                    if p.is_symlink() and p.resolve() == path
                ]
                # Validate file extension
                is_valid = path.suffix in self.allowed_extensions
                validation_errors = [] if is_valid else [f"Unsupported file extension: {path.suffix}"]
                metadata = FileMetadata(
                    path=path,
                    mime_type=mime_type or 'application/octet-stream',
                    size=stat.st_size,
                    created=datetime.fromtimestamp(stat.st_ctime),
                    modified=datetime.fromtimestamp(stat.st_mtime),
                    content_hash=self.compute_hash(path),
                    symlinks=symlinks,
                    is_valid=is_valid,
                    validation_errors=validation_errors
                )
                # Cache the result
                self.metadata_cache.set(cache_key, metadata)
                # Add to file watcher
                self._file_watchers[path] = time.time()
                return metadata
            except ContentError as e:
                raise e
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to get metadata for {path}: {e}", exc_info=True)
                raise ContentError(f"Failed to get metadata for {path}: {e}")
    @measure_performance
    def load_module(self, path: pathlib.Path) -> Optional[Any]:
        """Load a Python module from file with caching."""
        if path.suffix not in self.allowed_extensions:
            if self.logger:
                self.logger.warning(f"Skipping unsupported file extension: {path}")
            return None
        cache_key = str(path.absolute())
        # Check module cache
        cached_module = self.module_cache.get(cache_key)
        if cached_module:
            return cached_module
        with self._lock:
            try:
                # Get metadata first to validate file
                metadata = self.get_metadata(path)
                if not metadata.is_valid:
                    if self.logger:
                        self.logger.warning(f"Skipping invalid file: {path}, errors: {metadata.validation_errors}")
                    return None   
                module_name = f"content_{path.stem}"
                if path.suffix == '.py':
                    # Load as Python module
                    spec = importlib.util.spec_from_file_location(module_name, str(path))
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        setattr(module, '__metadata__', metadata)
                        # Execute the module
                        spec.loader.exec_module(module)
                        self.module_cache.set(cache_key, module)
                        return module
                else:
                    # Create a simple namespace for non-Python files
                    module = SimpleNamespace()
                    setattr(module, '__metadata__', metadata)
                    # Load content
                    content = self.get_content(path)
                    setattr(module, '__content__', content)
                    self.module_cache.set(cache_key, module)
                    return module
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to load module {path}: {e}", exc_info=True)
                return None
    @measure_performance
    def get_content(self, path: pathlib.Path) -> str:
        """Get file content with caching."""
        cache_key = f"content:{str(path.absolute())}"
        # Check content cache
        cached_content = self.content_cache.get(cache_key)
        if cached_content:
            return cached_content
        try:
            # Get metadata first to validate file
            metadata = self.get_metadata(path)
            if not metadata.is_valid:
                raise ContentError(f"Invalid file: {path}, errors: {metadata.validation_errors}")
            # Load content
            content = path.read_text(encoding='utf-8')
            # Cache the content
            self.content_cache.set(cache_key, content)
            return content
        except UnicodeDecodeError:
            # Try binary read for non-text files
            if self.logger:
                self.logger.warning(f"File appears to be binary, reading as bytes: {path}")
            return f"[Binary content, size: {path.stat().st_size} bytes]"
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to read content from {path}: {e}", exc_info=True)
            raise ContentError(f"Failed to read content from {path}: {e}")
    @measure_performance
    def scan_directory(self, directory: Optional[pathlib.Path] = None) -> Dict[str, FileMetadata]:
        """Scan directory recursively and cache metadata for valid files."""
        scan_dir = directory or self.root_dir
        results = {}
        try:
            for path in scan_dir.rglob('*'):
                if path.is_file() and path.suffix in self.allowed_extensions:
                    try:
                        metadata = self.get_metadata(path)
                        results[str(path.relative_to(self.root_dir))] = metadata
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Error processing {path}: {e}")
            if self.logger:
                self.logger.info(f"Scanned {len(results)} files in {scan_dir}")
            return results
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error scanning directory {scan_dir}: {e}", exc_info=True)
            raise ContentError(f"Error scanning directory {scan_dir}: {e}")
    def start_file_watcher(self, interval: int = 30) -> None:
        """Start a background thread to watch for file changes."""
        if self._watcher_thread and self._watcher_thread.is_alive():
            return  # Already running
        def watcher_loop():
            while not self._stop_event.is_set():
                self._check_file_changes()
                self._stop_event.wait(interval)
        self._stop_event.clear()
        self._watcher_thread = threading.Thread(target=watcher_loop, daemon=True)
        self._watcher_thread.start()
        if self.logger:
            self.logger.info(f"File watcher started with {interval}s interval")
    def stop_file_watcher(self) -> None:
        """Stop the file watcher thread."""
        if self._watcher_thread and self._watcher_thread.is_alive():
            self._stop_event.set()
            self._watcher_thread.join(timeout=5)
            if self.logger:
                self.logger.info("File watcher stopped")
    def _check_file_changes(self) -> None:
        """Check watched files for changes and invalidate caches."""
        with self._lock:
            paths_to_check = list(self._file_watchers.keys())
        for path in paths_to_check:
            try:
                if not path.exists():
                    # File was deleted
                    self._invalidate_file_caches(path)
                    with self._lock:
                        if path in self._file_watchers:
                            del self._file_watchers[path]
                    continue
                mtime = path.stat().st_mtime
                last_check = self._file_watchers.get(path, 0)
                if mtime > last_check:
                    # File was modified
                    self._invalidate_file_caches(path)
                    # Update last check time
                    with self._lock:
                        self._file_watchers[path] = time.time()
                    if self.logger:
                        self.logger.debug(f"Detected change in file: {path}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error checking file {path} for changes: {e}")
    def _invalidate_file_caches(self, path: pathlib.Path) -> None:
        """Invalidate all caches for a specific file."""
        abs_path = str(path.absolute())
        self.metadata_cache.invalidate(abs_path)
        self.content_cache.invalidate(f"content:{abs_path}")
        self.module_cache.invalidate(abs_path)
class RuntimeNamespace:
    """Hierarchical namespace manager with security controls."""
    def __init__(self, name: str = "root", parent: Optional['RuntimeNamespace'] = None, 
                 logger: Optional[logging.Logger] = None):
        self.name = name
        self.parent = parent
        self.children: Dict[str, 'RuntimeNamespace'] = {}
        self.content = SimpleNamespace()
        self.security_context: Optional[SecurityContext] = None
        self.frame_model: Optional[FrameModel] = None
        self.logger = logger
        self._lock = threading.RLock()
        self._created_at = datetime.now()
        self._last_accessed = datetime.now()
    @property
    def full_path(self) -> str:
        """Get full dot-separated namespace path."""
        return f"{self.parent.full_path}.{self.name}" if self.parent else self.name
    def add_child(self, name: str) -> 'RuntimeNamespace':
        """Add a child namespace with security check."""
        with self._lock:
            if self.security_context and not self.security_context.check_access(
                self.full_path, "write"
            ):
                raise SecurityError(f"Not authorized to add child to {self.full_path}")
            if name in self.children:
                raise NamespaceError(f"Namespace {name} already exists under {self.full_path}")
            child = RuntimeNamespace(name, self, self.logger)
            child.security_context = self.security_context
            child.frame_model = self.frame_model
            self.children[name] = child
            if self.logger:
                self.logger.debug(f"Created namespace {child.full_path}")
            return child
    def get_child(self, path: str) -> Optional['RuntimeNamespace']:
        """Get child namespace by dotted path with security check."""
        if not path:
            return self
        parts = path.split('.')
        current = self
        for part in parts:
            with current._lock:
                if current.security_context and not current.security_context.check_access(
                    current.full_path, "read"
                ):
                    raise SecurityError(f"Not authorized to access {current.full_path}")
                if part not in current.children:
                    return None
                current = current.children[part]
                current._last_accessed = datetime.now()
        return current
    def remove_child(self, name: str) -> bool:
        """Remove a child namespace with security check."""
        with self._lock:
            if self.security_context and not self.security_context.check_access(
                self.full_path, "delete"
            ):
                raise SecurityError(f"Not authorized to remove child from {self.full_path}")
            if name not in self.children:
                return False
            child_path = f"{self.full_path}.{name}"
            del self.children[name]
            if self.logger:
                self.logger.info(f"Removed namespace {child_path}")
            return True
    def embed_content(self, raw_content: str) -> None:
        """Embed content using FrameModel parsing with security check."""
        if self.security_context and not self.security_context.check_access(
            self.full_path, "write"
        ):
            raise SecurityError(f"Not authorized to write to {self.full_path}")
        if not self.frame_model:
            raise NamespaceError("No FrameModel configured")
        try:
            if not self.frame_model.validate_content(raw_content):
                raise ContentError("Invalid content format")
            parsed = self.frame_model.parse_content(raw_content)
            self.content.embedded_data = parsed
            self._last_accessed = datetime.now()
            if self.logger:
                self.logger.debug(f"Content embedded in {self.full_path}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to embed content in {self.full_path}: {e}")
            raise ContentError(f"Failed to embed content: {str(e)}")
    def retrieve_content(self) -> str:
        """Retrieve content with delimiters with security check."""
        if self.security_context and not self.security_context.check_access(
            self.full_path, "read"
        ):
            raise SecurityError(f"Not authorized to read from {self.full_path}")
        if not hasattr(self.content, 'embedded_data'):
            raise ContentError("No content embedded")
        if not self.frame_model:
            raise NamespaceError("No FrameModel configured") 
        self._last_accessed = datetime.now()
        return (self.frame_model.start_delimiter + 
                self.content.embedded_data + 
                self.frame_model.end_delimiter)
# ------------------
# Main Application
# ------------------
class ApplicationCore:
    """Central application controller."""
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig.from_env()
        self.logger = setup_logging(self.config)
        self.content_manager = ContentManager(self.config, self.logger)
        self.root_namespace = RuntimeNamespace(logger=self.logger)
        self.root_namespace.frame_model = FrameModel("{{", "}}")
        self.running = False
        self._scan_interval = 60  # seconds
        self._scan_thread = None
        self._shutdown_event = threading.Event()
        self.logger.info(f"Application initialized in {self.config.environment} mode")
    def create_security_context(self, user_id: str) -> SecurityContext:
        """Create appropriate security context based on user."""
        if user_id in self.config.admin_users:
            policy = AccessPolicy(
                level=AccessLevel.ADMIN,
                namespace_patterns=["*"],
                allowed_operations={"read", "write", "execute", "delete"}
            )
        else:
            policy = AccessPolicy(
                level=AccessLevel.READ,
                namespace_patterns=["public.*"],
                allowed_operations={"read"}
            )
        return SecurityContext(user_id, policy, self.logger)
    def _scan_worker(self):
        """Background worker to scan for content changes."""
        self.logger.info("Content scan worker started")
        while not self._shutdown_event.is_set():
            try:
                result = self.content_manager.scan_directory(incremental=True)
                self.logger.info(
                    f"Incremental scan complete: {result['loaded']} loaded, "
                    f"{result['modified']} modified, {result['failed']} failed"
                )
            except Exception as e:
                self.logger.error(f"Scan worker error: {e}")
            # Wait for next interval or shutdown
            self._shutdown_event.wait(self._scan_interval)
        self.logger.info("Content scan worker stopped")
    def start(self):
        """Start application services."""
        if self.running:
            return
        try:
            # Initial content scan
            self.logger.info("Starting initial content scan...")
            result = self.content_manager.scan_directory(incremental=False)
            self.logger.info(f"Initial scan complete: {result['loaded']} files loaded")
            # Start background services
            self._shutdown_event.clear()
            self._scan_thread = threading.Thread(
                target=self._scan_worker, 
                name="ContentScanWorker",
                daemon=True
            )
            self._scan_thread.start()
            self.running = True
            self.logger.info("Application started successfully")
        except Exception as e:
            self.logger.critical(f"Failed to start application: {e}")
            raise
    def stop(self):
        """Stop application services."""
        if not self.running:
            return
        else:
            self.logger.info("Stopping application...")
            self._shutdown_event.set()
            self._scan_thread.join()
            self.running = False
            self.logger.info("Application stopped successfully")
        try:
            self.content_manager.close()
        except: pass
        finally: self.content_manager = None; self.logger = None; self._shutdown_event = None; self._scan_thread = None
        return True



# old main()
def main():
    """Demonstration of core functionality."""
    # Setup configuration
    config = AppConfig(
        root_dir=pathlib.Path("."),
        log_level=logging.INFO,
        allowed_extensions={'.py', '.txt', '.md'},
        admin_users={'admin'}
    )
    
    # Initialize components
    logger = setup_logging(config)
    content_manager = ContentManager(config)
    
    # Example security setup
    admin_policy = AccessPolicy(
        level=AccessLevel.ADMIN,
        namespace_patterns=["*"],
        allowed_operations={"read", "write", "execute"}
    )
    security = SecurityContext("admin", admin_policy)
    
    # Example namespace setup
    root_ns = RuntimeNamespace()
    root_ns.security_context = security
    root_ns.frame_model = FrameModel("{{", "}}")
    
    # Example content embedding
    try:
        root_ns.embed_content("{{ This is test content }}")
        print("Embedded content:", root_ns.retrieve_content())
    except ValueError as e:
        logger.error(f"Content error: {e}")
    
    # Example file scanning
    try:
        result = content_manager.scan_directory()
        logger.info(f"Scanned {len(content_manager.module_cache)} files")
    except Exception as e:
        logger.error(f"Scan failed: {e}")
    
    # Example security check
    try:
        security.log_access("root", "read", True)
        print("Audit log:", security.get_audit_log())
    except Exception as e:
        logger.error(f"Security error: {e}")
class FrameModel(Generic[T, V, C]):
    """
    A frame model is a data structure that contains the data of a frame,
    representing a "measured reality" through delimited content.
    This notion bakes-in the notion of relativity and Markovian behavior..
    The frame model is a "first class citizen" in the sense
    that it can be used as a type, and can be used to create a new type.
    Like with WORD_SIZE, FrameModel(s) can scale and represent diverse data types,
    in theory any possible data type in the Architecture/Morphology.
    """
    def __init__(self, start_delimiter: str = "<<CONTENT>>", end_delimiter: str = "<<END_CONTENT>>"):
        self.start_delimiter = start_delimiter
        self.end_delimiter = end_delimiter
    def to_bytes(self) -> bytes:
        """Return the frame data as bytes, representing the extracted "measured reality"."""
        pass
    @measure_performance
    def parse_content(self, raw_content: str) -> str:
        """Parse content between delimiters, handling nested cases."""
        if not self.validate_content(raw_content):
            raise ContentError("Invalid content format: mismatched delimiters")
        # Extract content between first start and last end delimiter
        start_idx = raw_content.find(self.start_delimiter) + len(self.start_delimiter)
        end_idx = raw_content.rfind(self.end_delimiter)
        if start_idx < 0 or end_idx < 0 or start_idx >= end_idx:
            raise ContentError("Invalid content format: cannot extract content between delimiters")
        return raw_content[start_idx:end_idx].strip()
    def validate_content(self, raw_content: str) -> bool:
        """Validate content has balanced delimiters."""
        if not raw_content:
            return False
        start_count = raw_content.count(self.start_delimiter)
        end_count = raw_content.count(self.end_delimiter)
        # Basic validation: equal number of delimiters
        if start_count != end_count or start_count == 0:
            return False
        # Advanced validation: properly nested delimiters
        pos = 0
        depth = 0
        while pos < len(raw_content):
            if raw_content[pos:pos+len(self.start_delimiter)] == self.start_delimiter:
                depth += 1
                pos += len(self.start_delimiter)
            elif raw_content[pos:pos+len(self.end_delimiter)] == self.end_delimiter:
                depth -= 1
                if depth < 0:  # Closing delimiter before opening
                    return False
                pos += len(self.end_delimiter)
            else:
                pos += 1 
        return depth == 0

if __name__ == "__main__":
    main()
    import argparse
    parser = argparse.ArgumentParser(description="Content scanner application")
    parser.add_argument("-c", "--config", help="Path to configuration file")
    args = parser.parse_args()