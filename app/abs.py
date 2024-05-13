# abs.py abstract script
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass, field
from typing import List

class FrameModel(ABC):
   """
   Abstract base class representing a frame data structure.

   This class defines the abstract methods for serialization and deserialization of data.
   Concrete subclasses must implement these methods.

   Methods:
       to_bytes(): Return the frame data as bytes.
       to_pipe(pipe): Write the frame data to a named pipe.
       __delattr__(name): Perform platform-specific deletion of an attribute.
   """

   @abstractmethod
   def to_bytes(self) -> bytes:
       """Return the frame data as bytes."""
       pass

   @abstractmethod
   def to_pipe(self, pipe) -> None:
       """Write the frame data to a named pipe."""
       pass

   @abstractmethod
   def __delattr__(self, name: str) -> None:
       """Perform platform-specific deletion of an attribute."""
       pass

@dataclass(frozen=True, repr=False)
class SerialObject(FrameModel):
   """
   Concrete implementation of FrameModel demonstrating polymorphism.

   This class represents a serializable object with various attributes and methods
   for serialization and deserialization.

   Attributes:
       _id (int): Unique identifier for the object.
       _created_at (int): Timestamp when the object was created.
       _updated_at (int): Timestamp when the object was last updated.
       _deleted_at (int): Timestamp when the object was deleted (if applicable).
       _attributes (dict): Dictionary of additional attributes.
       _elements (List[str]): List of element names.
       _entities (List[str]): List of entity names/associations.

   Methods:
       id: Return the unique identifier of the object.
       to_bytes(): Return the object as bytes.
       to_pipe(pipe): Write the object to a named pipe.
   """

   _id: int = field(default_factory=lambda: uuid.uuid4().int)
   _created_at: int = field(default_factory=lambda: int(datetime.now().timestamp()))
   _updated_at: int = field(default_factory=lambda: int(datetime.now().timestamp()))
   _deleted_at: int = field(default_factory=lambda: int(datetime.now().timestamp()))
   _attributes: dict = field(default_factory=dict)
   _elements: List[str] = field(default_factory=list)
   _entities: List[str] = field(default_factory=list)

   @property
   def id(self) -> int:
       """Return the unique identifier of the object."""
       return self._id

   @abstractmethod
   def to_bytes(self) -> bytes:
       """Return the object as bytes."""
       pass

   @abstractmethod
   def to_pipe(self, pipe) -> None:
       """Write the object to a named pipe."""
       pass

@dataclass(frozen=True, repr=False)
class Element(FrameModel, ABC):
   """
   Abstract base class representing an element in the data model.

   This class defines the fundamental structure for various data model elements.
   It serves as an abstract interface, allowing for polymorphism and composition.

   Attributes:
       name (str): Name of the element.
       description (str): Description of the element.

   Methods:
       to_bytes(): Return the element data as bytes.
       to_str(): Return the element data as a string representation.
       dict(): Return a dictionary representation of the element.
   """

   name: str
   description: str

   @abstractmethod
   def to_bytes(self) -> bytes:
       """Return the element data as bytes."""
       pass

   @abstractmethod
   def to_str(self) -> str:
       """Return the element data as a string representation."""
       pass

   @abstractmethod
   def dict(self) -> dict:
       """Return a dictionary representation of the element."""
       pass

@dataclass(frozen=True, repr=False)
class Attribute(Element):
   """
   Class representing an attribute element.

   An attribute is a specific type of element with a defined name, description, and data type.
   It inherits from the abstract base class Element.

   Attributes:
       data_type (str): Data type of the attribute.

   Raises:
       ValueError: If an invalid data type is provided.
   """

   data_type: str

   _ALLOWED_TYPES = {"TEXT", "INTEGER", "REAL", "BLOB", "VARCHAR", "BOOLEAN", "UFS", "VECTOR", "TIMESTAMP", "EMBEDDING"}

   def __post_init__(self):
       if self.data_type not in self._ALLOWED_TYPES:
           raise ValueError(f"Invalid data type: {self.data_type}")

@dataclass(frozen=True, repr=False)
class Entity(Element):
   """
   Class representing an entity element.

   Entity inherits from Element and contains a list of Element instances.
   This allows Entity objects to contain Attribute objects and any other objects that are subclasses of Element.

   Attributes:
       elements (List[Element]): List of elements contained within the entity.
       serial_model (ConcreteSerialModel): ConcreteSerialModel for serialization.
   """

   elements: List[Element] = field(default_factory=list)
   serial_model: SerialObject = None

   def __str__(self) -> str:
       """Return a user-friendly string representation of the Entity object."""
       return f"Name: {self.name}\nDescription: {self.description}"

   def to_str(self) -> str:
       """Return a string representation of the Entity object, including its elements."""
       return '\n<im_start>'.join([e.to_str() for e in self.elements]) + '\n<im_end>\n'

