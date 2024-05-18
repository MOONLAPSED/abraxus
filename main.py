from src.app import AtomDataclass, T, FormalTheory, repl
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Generic, TypeVar, Any


# Define the non-linear wave equation (simplified example)
def soliton_update(u, du):
    c = 1.0  # Speed of wave
    return 2 * u - du + c * (np.roll(u, 1) - 2*u + np.roll(u, -1))

def main():
    T = TypeVar('T')
    # Initial setup: a soliton-like pulse
    N = 100         # Number of lattice points
    u = np.zeros(N) # Main state array
    du = np.zeros(N)# Previous state array

    # Create a soliton-like initial condition
    u[N//2 - 5:N//2 + 5] = np.hanning(10)

    # Store states for visualization
    states = [u.copy()]

    # Run the simulation for a number of steps
    steps = 100
    for _ in range(steps):
        u_new = soliton_update(u, du)
        du = u
        u = u_new
        states.append(u.copy())

    # Visualization
    plt.imshow(states, extent=[0, N, 0, steps], aspect='auto', cmap='viridis')
    plt.colorbar(label='Amplitude')
    plt.xlabel('Lattice point')
    plt.ylabel('Time step')
    plt.title('Soliton Propagation Simulation')
    plt.show()


@dataclass
class FormalTheory(Generic[T]):
    reflexivity: Callable[[T], bool] = lambda x: x == x
    symmetry: Callable[[T, T], bool] = lambda x, y: x == y
    transitivity: Callable[[T, T, T], bool] = lambda x, y, z: (x == y) and (y == z) and (x == z)
    transparency: Callable[[Callable[..., T], T, T], T] = lambda f, x, y: f(True, x, y) if x == y else None
    case_base: Dict[str, Callable[[T, T], T]] = field(default_factory=dict)
    
    def __post_init__(self):
        self.case_base = {
            '⊤': lambda x, _: x,
            '⊥': lambda _, y: y,
            'a': self.if_else_a
        }

    def if_else(self, a: bool, x: T, y: T) -> T:
        return x if a else y

    def if_else_a(self, x: T, y: T) -> T:
        return self.if_else(True, x, y)

    def compare(self, atoms: List[Any]) -> bool:
        if not atoms:
            return False
        comparison = [self.symmetry(atoms[0], atoms[i]) for i in range(1, len(atoms))]
        return all(comparison)

    def encode(self) -> bytes:
        # Example encoding for formal theory properties
        return str(self.case_base).encode()

    def decode(self, data: bytes) -> None:
        # Example decoding for formal theory properties
        pass

    def execute(self, *args, **kwargs) -> Any:
        # Placeholder
        pass

    def __repr__(self):
        case_base_repr = {
            key: (value.__name__ if callable(value) else value)
            for key, value in self.case_base.items()
        }
        return (f"FormalTheory(\n"
                f"  reflexivity={self.reflexivity.__name__},\n"
                f"  symmetry={self.symmetry.__name__},\n"
                f"  transitivity={self.transitivity.__name__},\n"
                f"  transparency={self.transparency.__name__},\n"
                f"  case_base={case_base_repr}\n"
                f")")

def reflexivity(x: Any) -> bool:
    return x == x

def symmetry(x: Any, y: Any) -> bool:
    return x == y

if __name__ == "__main__":
    main()
    # repl()
