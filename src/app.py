from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Generic, TypeVar, Optional
import logging
import sys
import struct
from argparse import ArgumentParser
from main import Atom

class AtomicBot(Atom, ABC):
    """
    Abstract base class for AtomicBot implementations.
    """
    @abstractmethod
    def send_event(self, event: Dict[str, Any]) -> None:
        """
        Send an event to the AtomicBot implementation.
        Args:
            event (Dict[str, Any]): The event to send, following the event format specified in the AtomicBot Standard.
        """
        pass

    @abstractmethod
    def handle_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an action request from an AtomicBot application.
        Args:
            action (Dict[str, Any]): The action request, following the action format specified in the AtomicBot Standard.
        Returns:
            Dict[str, Any]: The action response, following the action response format specified in the AtomicBot Standard.
        """
        pass

    @abstractmethod
    def start_http_server(self, host: str, port: int, access_token: Optional[str] = None, event_enabled: bool = False, event_buffer_size: int = 100) -> None:
        """
        Start an HTTP server for handling AtomicBot communication.
        Args:
            host (str): The IP address to listen on.
            port (int): The port to listen on.
            access_token (Optional[str]): The access token for authentication (if required).
            event_enabled (bool): Whether event polling should be enabled.
            event_buffer_size (int): The size of the event buffer for event polling.
        """
        pass

    @abstractmethod
    def start_websocket_server(self, host: str, port: int, access_token: Optional[str] = None) -> None:
        """
        Start a WebSocket server for handling AtomicBot communication.
        Args:
            host (str): The IP address to listen on.
            port (int): The port to listen on.
            access_token (Optional[str]): The access token for authentication (if required).
        """
        pass

    @abstractmethod
    def start_websocket_client(self, url: str, access_token: Optional[str] = None, reconnect_interval: Optional[int] = None) -> None:
        """
        Connect to a WebSocket endpoint for handling AtomicBot communication.
        Args:
            url (str): The WebSocket URL to connect to.
            access_token (Optional[str]): The access token for authentication (if required).
            reconnect_interval (Optional[int]): The interval (in seconds) to reconnect if the connection is lost.
        """
        pass

    @abstractmethod
    def set_webhook(self, url: str, access_token: Optional[str] = None, timeout: Optional[int] = None) -> None:
        """
        Set the webhook URL for receiving events from the AtomicBot implementation.
        Args:
            url (str): The webhook URL to receive events.
            access_token (Optional[str]): The access token for authentication (if required).
            timeout (Optional[int]): The timeout (in seconds) for webhook requests.
        """
        pass

class EventBase(Atom, ABC):
    """
    Abstract base class for Events, defining the common interface and methods.
    """
    @abstractmethod
    def encode(self) -> bytes:
        pass
    @abstractmethod
    def decode(self, data: bytes) -> None:
        pass
    @abstractmethod
    def to_dataclass(self) -> Dict[str, Any]:
        pass

class ActionBase(Atom, ABC):
    """
    Abstract base class for Actions, defining the common interface and methods.
    """
    @abstractmethod
    def encode(self) -> bytes:
        pass
    @abstractmethod
    def decode(self, data: bytes) -> None:
        pass
    @abstractmethod
    def to_dataclass(self) -> Dict[str, Any]:
        pass

class Event(EventBase):
    """
    A class representing an AtomicBot event.
    """
    def __init__(self, event_id: str, event_type: str, detail_type: Optional[str] = None, message: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> None:
        self.event_data = {
            "id": event_id,
            "type": event_type,
            "detail_type": detail_type,
            "message": message or [],
            **kwargs
        }
    def encode(self) -> bytes:
        return str(self.event_data).encode('utf-8')
    def decode(self, data: bytes) -> None:
        self.event_data = eval(data.decode('utf-8'))
    def to_dataclass(self) -> Dict[str, Any]:
        return self.event_data

class Action(ActionBase):
    """
    A class representing an AtomicBot action request.
    """
    def __init__(self, action_name: str, params: Optional[Dict[str, Any]] = None, self_info: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self.action_data = {
            "action": action_name,
            "params": params or {},
            "self": self_info or {},
            **kwargs
        }
    def encode(self) -> bytes:
        return str(self.action_data).encode('utf-8')
    def decode(self, data: bytes) -> None:
        self.action_data = eval(data.decode('utf-8'))
    def to_dataclass(self) -> Dict[str, Any]:
        return self.action_data

class ActionResponse(ActionBase):
    """
    A class representing an AtomicBot action response.
    """
    def __init__(self, action_name: str, status: str, retcode: int, data: Optional[Dict[str, Any]] = None, message: Optional[str] = None, **kwargs: Any) -> None:
        self.response_data = {
            "resp": action_name,
            "status": status,
            "retcode": retcode,
            "data": data or {},
            "message": message,
            **kwargs
        }
    def encode(self) -> bytes:
        return str(self.response_data).encode('utf-8')
    def decode(self, data: bytes) -> None:
        self.response_data = eval(data.decode('utf-8'))
    def to_dataclass(self) -> Dict[str, Any]:
        return self.response_data
