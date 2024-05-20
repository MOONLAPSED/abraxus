from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional, Generic, TypeVar
import struct

# Type Variables
T = TypeVar('T')

# Abstract Base Class for all Atom types
class Atom(ABC):
    @abstractmethod
    def encode(self) -> bytes:
        pass

    @abstractmethod
    def decode(self, data: bytes) -> None:
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def to_dataclass(self):
        pass

@dataclass
class ParseTreeAtom(Atom):
    label: str
    children: List['ParseTreeAtom'] = field(default_factory=list)

    def encode(self) -> bytes:
        # Implement encoding if needed
        pass

    def decode(self, data: bytes) -> None:
        # Implement decoding if needed
        pass

    def execute(self, *args, **kwargs) -> Any:
        # Implement execution if needed
        pass

    def __repr__(self):
        return self.to_bnf(0)

    def to_dataclass(self):
        return self

    def to_bnf(self, indent: int = 0) -> str:
        indentation = '  ' * indent
        bnf_repr = f"{indentation}<{self.label}>\n"
        for child in self.children:
            if isinstance(child, ParseTreeAtom):
                bnf_repr += child.to_bnf(indent + 1)
            else:
                bnf_repr += f"{indentation}  {child}\n"
        return bnf_repr

    class ParseTreeAtom:
        def __init__(self, label, value=None, children=None):
            self.label = label
            self.value = value
            self.children = children if children else []  # Ensure children is always a list


    def to_string(self, indent=""):
        """
        Returns a string representation of the ParseTreeAtom and its children.

        Args:
            indent (str, optional): The current indentation level. Defaults to "".

        Returns:
            str: The string representation of the syntax tree.
        """
        result = f"{indent}{self.label}"
        if self.value:
            result += f"({self.value})"
        if self.children:
            result += "\n"
            for child in self.children:
                result += child.to_string(indent + "  ")
        return result

def generate_bnf(obj, class_name=None, indent=0):
    if class_name is None:
        class_name = type(obj).__name__

    bnf = f"{' ' * indent}<{class_name}>"
    if hasattr(obj, "data_type"):
        bnf += f" ::= <data_type> {obj.data_type}"

    if hasattr(obj, "value"):
        value_bnf = generate_value_bnf(obj.value, indent + 2)
        bnf += f" <value> {value_bnf}"

    if hasattr(obj, "children"):
        bnf += " ::="
        for child in obj.children:
            child_bnf = generate_bnf(child, indent=indent + 2)
            bnf += f"\n{child_bnf}"

    return bnf

def generate_value_bnf(value: Any, indent=0):
    if isinstance(value, (str, int, float, bool)):
        return f"{' ' * indent}\"{value}\""
    elif isinstance(value, list):
        bnf = f"{' ' * indent}::= "
        for item in value:
            item_bnf = generate_value_bnf(item, indent + 2)
            bnf += f"\n{item_bnf}"
        return bnf
    elif isinstance(value, dict):
        bnf = f"{' ' * indent}::= "
        for key, val in value.items():
            key_bnf = generate_value_bnf(key, indent + 2)
            val_bnf = generate_value_bnf(val, indent + 2)
            bnf += f"\n{key_bnf} = {val_bnf}"
        return bnf
    else:
        return f"{' ' * indent}{value}"


@dataclass
class AtomDataclass(Generic[T], Atom):
    value: T
    data_type: str = field(init=False)
    
    def __post_init__(self):
        type_map = {
            'str': 'string',
            'int': 'integer',
            'float': 'float',
            'bool': 'boolean',
            'list': 'list',
            'dict': 'dictionary'
        }
        data_type_name = type(self.value).__name__
        object.__setattr__(self, 'data_type', type_map.get(data_type_name, 'unsupported'))

    def encode(self) -> bytes:
        data_type_bytes = self.data_type.encode('utf-8')
        data_bytes = self._encode_data()
        header = struct.pack('!I', len(data_type_bytes))
        return header + data_type_bytes + data_bytes

    def _encode_data(self) -> bytes:
        if self.data_type == 'string':
            return self.value.encode('utf-8')
        elif self.data_type == 'integer':
            return struct.pack('!q', self.value)
        elif self.data_type == 'float':
            return struct.pack('!d', self.value)
        elif self.data_type == 'boolean':
            return struct.pack('?', self.value)
        elif self.data_type == 'list':
            return b''.join([AtomDataclass(element)._encode_data() for element in self.value])
        elif self.data_type == 'dictionary':
            return b''.join(
                [AtomDataclass(key)._encode_data() + AtomDataclass(value)._encode_data() for key, value in self.value.items()]
            )
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")

    def decode(self, data: bytes) -> None:
        header_length = struct.unpack('!I', data[:4])[0]
        data_type_bytes = data[4:4 + header_length]
        data_type = data_type_bytes.decode('utf-8')
        data_bytes = data[4 + header_length:]
        
        type_map_reverse = {
            'string': 'str',
            'integer': 'int',
            'float': 'float',
            'boolean': 'bool',
            'list': 'list',
            'dictionary': 'dict'
        }

        if data_type == 'string':
            value = data_bytes.decode('utf-8')
        elif data_type == 'integer':
            value = struct.unpack('!q', data_bytes)[0]
        elif data_type == 'float':
            value = struct.unpack('!d', data_bytes)[0]
        elif data_type == 'boolean':
            value = struct.unpack('?', data_bytes)[0]
        elif data_type == 'list':
            value = []
            offset = 0
            while offset < len(data_bytes):
                element = AtomDataclass(None)
                element_size = element.decode(data_bytes[offset:])
                value.append(element.value)
                offset += element_size
        elif data_type == 'dictionary':
            value = {}
            offset = 0
            while offset < len(data_bytes):
                key = AtomDataclass(None)
                key_size = key.decode(data_bytes[offset:])
                offset += key_size
                val = AtomDataclass(None)
                value_size = val.decode(data_bytes[offset:])
                offset += value_size
                value[key.value] = val.value
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        self.value = value
        object.__setattr__(self, 'data_type', type_map_reverse.get(data_type, 'unsupported'))

    def execute(self, *args, **kwargs) -> Any:
        pass

    def __repr__(self):
        return f"AtomDataclass(id={id(self)}, value={self.value}, data_type='{self.data_type}')"

    def to_dataclass(self):
        return self
    
    # Overloaded operators
    def __add__(self, other):
        return AtomDataclass(self.value + other.value)

    def __sub__(self, other):
        return AtomDataclass(self.value - other.value)

    def __mul__(self, other):
        return AtomDataclass(self.value * other.value)

    def __truediv__(self, other):
        return AtomDataclass(self.value / other.value)

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value

@dataclass
class FormalTheory(Atom, Generic[T]):
    reflexivity: Callable[[T], bool] = lambda x: x == x
    symmetry: Callable[[T, T], bool] = lambda x, y: x == y
    transitivity: Callable[[T, T, T], bool] = lambda x, y, z: (x == y) and (y == z) and (x == z)
    transparency: Callable[[Callable[..., T], T, T], T] = lambda f, x, y: f(True, x, y) if x == y else None
    case_base: Dict[str, Callable[..., bool]] = field(default_factory=dict)
    
    def __post_init__(self):
        self.case_base = {
            '⊤': lambda x, _: x,
            '⊥': lambda _, y: y,
            'a': self.if_else_a,
            '¬': lambda a: not a,
            '∧': lambda a, b: a and b,
            '∨': lambda a, b: a or b,
            '→': lambda a, b: (not a) or b,
            '↔': lambda a, b: (a and b) or (not a and not b),
            '¬∨': lambda a, b: not (a or b),  # NOR operation
            '¬∧': lambda a, b: not (a and b),  # NAND operation
            'contrapositive': self.contrapositive
        }
    
    def if_else(self, a: bool, x: T, y: T) -> T:
        return x if a else y

    def if_else_a(self, x: T, y: T) -> T:
        return self.if_else(True, x, y)

    def contrapositive(self, a: bool, b: bool) -> bool:
        return (not b) or (not a)
    
    def compare(self, atoms: List[AtomDataclass[T]]) -> bool:
        if not atoms:
            return False
        comparison = [self.symmetry(atoms[0].value, atoms[i].value) for i in range(1, len(atoms))]
        return all(comparison)
    
    def encode(self) -> bytes:
        return str(self.case_base).encode()

    def decode(self, data: bytes) -> None:
        # Example decoding for formal theory properties
        pass

    def execute(self, *args, **kwargs) -> Any:
        pass

    def to_dataclass(self):
        return super().to_dataclass()
    
    def __repr__(self):
        case_base_repr = {
            key: (value.__name__ if callable(value) else value)
            for key, value in self.case_base.items()
        }
        return f"""FormalTheory(
    reflexivity={self.reflexivity.__name__},
    symmetry={self.symmetry.__name__},
    transitivity={self.transitivity.__name__},
    transparency={self.transparency.__name__},
    case_base={case_base_repr}
    )"""




def create_digit_atom(label, value):
    """
    Creates a ParseTreeAtom for a single digit.

    Args:
        label (str): The label for the atom (e.g., "digit").
        value (str): The value of the digit (e.g., "9").

    Returns:
        ParseTreeAtom: The created ParseTreeAtom object.
    """
    if isinstance(value, int):
        digits = [ParseTreeAtom("digit", str(d)) for d in str(value)]
        return ParseTreeAtom(label="integer", children=digits)
    return ParseTreeAtom(label=label, children=[ParseTreeAtom(label=str(value))])

def create_integer_atom(value):
    """
    Creates a ParseTreeAtom for an integer.

    Args:
        value (int or float): The value of the integer.

    Returns:
        ParseTreeAtom: The created ParseTreeAtom object representing the integer.
    """
    if isinstance(value, int):
        # Handle integer case
        digits = [create_digit_atom("digit", str(d)) for d in str(value)]  # Create digit atoms for each digit
        return ParseTreeAtom(label="integer", children=digits)
    elif isinstance(value, float):
        # Handle float case (assuming dot notation)
        integer_part_str = str(int(value))  # Extract integer part as string
        decimal_part_str = f".{str(value)[len(integer_part_str):]}"  # Extract decimal part as string with dot
        integer_part = create_integer_atom(int(integer_part_str))  # Create ParseTreeAtom for integer part
        decimal_part = create_digit_atom(".", decimal_part_str)  # Create ParseTreeAtom for decimal part
        return ParseTreeAtom(label="floating-point", children=[integer_part, decimal_part])
    else:
        raise TypeError("create_integer_atom only accepts int or float values")

def create_floating_point(value):
    """
    Creates a ParseTreeAtom for a floating-point number.

    Args:
        value (float): The floating-point value.

    Returns:
        ParseTreeAtom: The created ParseTreeAtom representing the floating-point number.
    """
    if not isinstance(value, float):
        raise TypeError("create_floating_point only accepts float values")

    integer_part_str = str(int(value))
    decimal_part_str = f".{str(value)[len(integer_part_str):]}"

    integer_part = create_integer_atom(int(integer_part_str))
    decimal_part = create_digit_atom(".", decimal_part_str)

    return ParseTreeAtom(label="floating-point", children=[integer_part, decimal_part])

# Example usage:
if __name__ == "__main__":
    # Constructing a parse tree based on the provided image
    digit_9 = ParseTreeAtom(label="digit", children=[ParseTreeAtom(label="9")])
    digit_3 = ParseTreeAtom(label="digit", children=[ParseTreeAtom(label="3")])
    integer_93 = ParseTreeAtom(label="integer", children=[digit_9, digit_3])

    digit_4 = ParseTreeAtom(label="digit", children=[ParseTreeAtom(label="4")])
    integer_4 = ParseTreeAtom(label="integer", children=[digit_4])

    floating_point = ParseTreeAtom(label="floating-point", children=[
        integer_93, 
        ParseTreeAtom(label="."), 
        integer_4
    ])

    # Printing the parse tree
    #floating_point.children[1].value = "."
    integer_93 = create_integer_atom(93)  # Creates ParseTreeAtom for integer 93
    floating_point_3_14 = create_integer_atom(3.14)  # Creates ParseTreeAtom for float 3.14

    # Other AtomDataclass examples
    atom1 = AtomDataclass(value=10)
    atom2 = AtomDataclass(value=20)
    atom3 = atom1 + atom2

    print(atom1)
    print(atom2)
    print(f"atom1 + atom2 = {atom3}")

    # Encoding and decoding example
    encoded_atom3 = atom3.encode()
    print(f"Encoded atom3: {encoded_atom3}")

    # Create a new atom and decode into it
    decoded_atom3 = AtomDataclass(value=0)
    decoded_atom3.decode(encoded_atom3)
    print(f"Decoded atom3: {decoded_atom3}")

    # FormalTheory example usage
    formal_theory = FormalTheory[int]()
    print(formal_theory)
    encoded_theory = formal_theory.encode()
    print(f"Encoded theory: {encoded_theory}")
    compare_result = formal_theory.compare([atom1, atom2])
    contrapositive_result = formal_theory.contrapositive(True, False)
    print(f"Comparison result: {compare_result}")
    print(f"Contrapositive result: {contrapositive_result}")

    # Print BNF representation
    print(floating_point.to_bnf())
    print(integer_93.to_bnf())

    # Create a ParseTreeAtom
    digit_9 = ParseTreeAtom(label="digit", children=[ParseTreeAtom(label="9")])
    digit_3 = ParseTreeAtom(label="digit", children=[ParseTreeAtom(label="3")])
    integer_93 = ParseTreeAtom(label="integer", children=[digit_9, digit_3])

    # Generate BNF for ParseTreeAtom
    print(generate_bnf(integer_93))

    # Create an AtomDataclass
    atom = AtomDataclass(value=[1, 2, {"key": "value"}])

    # Generate BNF for AtomDataclass
    print(generate_bnf(atom))
