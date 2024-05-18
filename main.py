from src.app import AtomDataclass, T, FormalTheory, repl
import functools
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Generic, TypeVar, Any
from dataclasses import dataclass, field
from typing import Callable, Dict, List, TypeVar, Any, Tuple



T = TypeVar('T')

@dataclass
class FormalTheory(Generic[T]):
    name: str = "Generic Theory"
    reflexivity: Callable[[T], bool] = lambda x: x == x
    symmetry: Callable[[T, T], bool] = lambda x, y: x == y
    transitivity: Callable[[T, T, T], bool] = lambda x, y, z: (x == y) and (y == z) and (x == z)
    transparency: Callable[[Callable[..., T], T, T], T] = lambda f, x, y: f(True, x, y) if x == y else None
    case_base: Dict[str, Callable[[T, T], T]] = field(default_factory=dict)
    states: List[np.ndarray] = field(default_factory=list)
    theory: Callable[[np.ndarray, Tuple], np.ndarray] = None
    anti_theory: Callable[[np.ndarray, Tuple], np.ndarray] = None

    def __post_init__(self):
        self.case_base = {
            '⊤': lambda x, _: x,  # Tautology: always true
            '⊥': lambda _, y: y,  # Contradiction: always false
            'Compare': self.compare
        }

    def if_else(self, a: bool, x: T, y: T) -> T:
        return x if a else y

    def compare(self, atoms: List[Any]) -> bool:
        if not atoms:
            return False
        comparison = [self.symmetry(atoms[0], atoms[i]) for i in range(1, len(atoms))]
        return all(comparison)

    def execute_theory(self, input_state: np.ndarray, *theory_params) -> np.ndarray:
        if self.theory:
            return self.theory(input_state, *theory_params)
        return input_state

    def execute_anti_theory(self, input_state: np.ndarray, *anti_theory_params) -> np.ndarray:
        if self.anti_theory:
            return self.anti_theory(input_state, *anti_theory_params)
        return input_state

    def run_simulation(self, steps: int, initial_state: np.ndarray, theory_params: Tuple = (), anti_theory_params: Tuple = ()):
        state = initial_state.copy()
        self.states.append(state.copy())
        for _ in range(steps):
            state = self.execute_theory(state, *theory_params)
            anti_state = self.execute_anti_theory(state, *anti_theory_params)
            # Combining theory and anti-theory for refined state
            state = 0.5 * (state + anti_state)
            self.states.append(state.copy())

    def plot_results(self):
        steps = len(self.states)
        states_array = np.array(self.states)
        plt.imshow(states_array, extent=[0, states_array.shape[1], 0, steps], aspect='auto', cmap='viridis')
        plt.colorbar(label='Amplitude')
        plt.xlabel('Lattice point')
        plt.ylabel('Time step')
        plt.title('Theory/Anti-Theory Simulation')
        plt.show()

# Example Usage
def example_theory(state: np.ndarray, c: float = 1.0) -> np.ndarray:
    return state + c * (np.roll(state, -1) - 2 * state + np.roll(state, 1))

def example_anti_theory(state: np.ndarray, c: float = 1.0) -> np.ndarray:
    return state - c * (np.roll(state, -1) - 2 * state + np.roll(state, 1))

initial_state = np.zeros(100)
initial_state[45:55] = np.hanning(10)

formal_theory = FormalTheory(theory=example_theory, anti_theory=example_anti_theory)
formal_theory.run_simulation(steps=100, initial_state=initial_state, theory_params=(1.0,), anti_theory_params=(1.0,))
formal_theory.plot_results()
