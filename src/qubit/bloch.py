"""The idea is to use constrain satisfaction, matrixes, and a limited resolution
space (default 2^8) to preform Newton lambda-calculus. We can use logical theories
and test them and string them together to create syntax and emergence representing
various conserved quantities as-yet not-understood which I posit are quantum
entanglment-related and also involve language."""
import math
import random
import struct
import turtle
import time

class ComplexNumber:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return ComplexNumber(self.real + other.real, self.imag + other.imag)

    def __truediv__(self, scalar):
        return ComplexNumber(self.real / scalar, self.imag / scalar)

    def __abs__(self):
        return math.sqrt(self.real**2 + self.imag**2)

    def conjugate(self):
        return ComplexNumber(self.real, -self.imag)

    def __repr__(self):
        return f"{self.real:.2f} + {self.imag:.2f}i"

class ComplexQubit:
    def __init__(self, alpha_real_bytes, alpha_imag_bytes):
        self.alpha = ComplexNumber(self._bytes_to_float(alpha_real_bytes),
                                   self._bytes_to_float(alpha_imag_bytes))
        self._normalize()

    def _bytes_to_float(self, byte_pair):
        return int.from_bytes(byte_pair, 'big', signed=True) / 32768

    def _normalize(self):
        norm = abs(self.alpha)
        self.alpha = self.alpha / norm  # Always normalize

    @property
    def beta(self):
        beta_real = math.sqrt(1 - (self.alpha.real**2 + self.alpha.imag**2))
        return ComplexNumber(beta_real, 0)

    def measure(self):
        prob_zero = abs(self.alpha)**2
        return 0 if random.random() < prob_zero else 1

    def apply_hadamard(self):
        new_alpha = (self.alpha + self.beta) / math.sqrt(2)
        new_real_bytes = int(new_alpha.real * 32768).to_bytes(2, 'big', signed=True)
        new_imag_bytes = int(new_alpha.imag * 32768).to_bytes(2, 'big', signed=True)
        return ComplexQubit(new_real_bytes, new_imag_bytes)

    def bloch_coordinates(self):
        theta = 2 * math.acos(abs(self.alpha))
        phi = math.atan2(self.beta.imag, self.beta.real) - math.atan2(self.alpha.imag, self.alpha.real)
        return (math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta))

def visualize_bloch_sphere(qubits):
    screen = turtle.Screen()
    screen.title("Qubit States on Bloch Sphere")
    screen.setup(width=800, height=800)

    t = turtle.Turtle()
    t.speed(0)
    t.hideturtle()

    # Draw Bloch sphere
    t.penup()
    t.goto(0, -200)
    t.pendown()
    t.circle(200)

    t.penup()
    t.goto(0, 0)
    t.pendown()
    t.circle(200, steps=36)

    t.penup()
    t.goto(0, 0)
    t.pendown()
    t.circle(200, steps=18)

    # Draw axes
    t.penup()
    t.goto(0, 0)
    t.pendown()
    t.goto(0, 200)
    t.goto(0, -200)
    t.penup()
    t.goto(0, 0)
    t.pendown()
    t.goto(200, 0)
    t.goto(-200, 0)

    # Plot qubit states
    for qubit in qubits:
        x, y, z = qubit.bloch_coordinates()
        t.penup()
        t.goto(x * 200, y * 200)
        t.dot(10, "red")

    time.sleep(5)
    turtle.bye()

# Example usage
qubits = [
    ComplexQubit((0,0), (0,0)),  # |0⟩
    ComplexQubit((255,255), (0,0)),  # |1⟩
    ComplexQubit((181,0), (0,0)),  # |+⟩
    ComplexQubit((181,0), (181,0)),  # |+i⟩
]

visualize_bloch_sphere(qubits)

# Demonstrate measurement
q = ComplexQubit((181,0), (0,0))  # |+⟩ state
measurements = [q.measure() for _ in range(1000)]
print(f"Measurement results for |+⟩: {sum(measurements)/len(measurements):.2f} ones")

# Demonstrate Hadamard gate
q = ComplexQubit((0,0), (0,0))  # |0⟩ state
q_after_h = q.apply_hadamard()
print(f"|0⟩ after Hadamard: α={q_after_h.alpha}, β={q_after_h.beta}")
