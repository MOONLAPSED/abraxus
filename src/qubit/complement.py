import math

def byte_to_bloch(byte_value):
    # Map 0-255 to -π to π
    theta = (byte_value / 128 - 1) * math.pi
    # Calculate φ (always 0 in this simple mapping)
    phi = 0
    # Calculate Bloch sphere coordinates
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)
    return x, y, z

def byte_to_qubit_state(byte_value):
    # Map 0-255 to 0-π
    theta = (byte_value / 255) * math.pi
    # Calculate α and β
    alpha = math.cos(theta/2)
    beta = math.sin(theta/2)
    return alpha, beta

# Example usage
print("Bloch coordinates:")
print("  |0⟩:", byte_to_bloch(0))
print("  |1⟩:", byte_to_bloch(255))
print("  |+⟩:", byte_to_bloch(128))
print("\nQubit states (α, β):")
print("  |0⟩:", byte_to_qubit_state(0))
print("  |1⟩:", byte_to_qubit_state(255))
print("  |+⟩:", byte_to_qubit_state(128))

# Demonstrate complementarity
test_value = 42
complement = 255 - test_value
print(f"\nComplementarity test for {test_value}:")
print(f"  {test_value}:   {byte_to_bloch(test_value)}")
print(f"  {complement}: {byte_to_bloch(complement)}")