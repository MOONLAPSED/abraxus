# main.py
import sys
from src.atom import DataUnit
from src.ops import DefaultOperations, addition, subtraction, multiplication, division, exponentiation
from src.turing_tape import TuringTape

def process_command(command: str, tape: TuringTape) -> str:
    if command.startswith("read"):
        return str(tape.read())
    elif command.startswith("write "):
        data = command.split(" ", 1)[1]
        tape.write(data)
        return f"Written: {data}"
    elif command == "move_right":
        tape.move_right()
        return "Moved right"
    elif command == "move_left":
        tape.move_left()
        return "Moved left"
    elif command == "show_tape":
        return str(tape)
    else:
        return "Command not recognized."
def kernel_agent(prompt: str, tape: TuringTape) -> str:
    # Split prompt into separate commands
    commands = prompt.split(";")
    responses = []

    for command in commands:
        response = process_command(command.strip(), tape)
        responses.append(response)
    
    # Generate final response
    final_response = "\n".join(responses)
    return final_response

# Read prompt from command-line arguments and process it
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <command_string>")
        sys.exit(1)
    
    prompt_input = " ".join(sys.argv[1:])
    tape = TuringTape()  # Initialize TuringTape
    result = kernel_agent(prompt_input, tape)
    
    # Output the result to STDOUT
    print(result)

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