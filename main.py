# main.py

from src.atom import DataUnit
from src.ops import DefaultOperations, addition, subtraction, multiplication, division, exponentiation

if __name__ == "__main__":
    string_data = DataUnit("Hello, World!")
    binary_data = DataUnit(b'\x12\x34\x56\x78')
    embedding_data = DataUnit([0.1, 0.2, 0.3])
    token_data = DataUnit([1, 0, 1, 1])
    
    print("String DataUnit:", string_data)
    print("Binary DataUnit:", binary_data)
    print("Embedding DataUnit:", embedding_data)
    print("Token DataUnit:", token_data)

    print("String DataUnit to base64:", string_data.to_base64())
    print("Binary DataUnit to hex:", binary_data.to_hex())
    
    # Demonstrate bitwise operations
    binary_data2 = DataUnit(b'\xff\x00\xff\x00')
    print("Binary DataUnit AND operation:", binary_data & binary_data2)
    print("Binary DataUnit OR operation:", binary_data | binary_data2)

    # Arithmetic operations example
    print(f"Addition: 5 + 3 = {addition(5, 3)}")
    print(f"Subtraction: 5 - 3 = {subtraction(5, 3)}")
    print(f"Multiplication: 5 * 3 = {multiplication(5, 3)}")
    print(f"Division: 5 / 3 = {division(5, 3)}")
    print(f"Exponentiation: 5 ** 3 = {exponentiation(5, 3)}")