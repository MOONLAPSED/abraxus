# src/turing_tape.py

from .atom import DataUnit
from typing import List, Union

class TuringTape:
    def __init__(self, initial_size: int = 10):
        self.tape: List[DataUnit] = [DataUnit("") for _ in range(initial_size)]  # Initialize tape with empty DataUnits
        self.cursor = 0  # Start at the beginning of the tape

    def expand_tape(self):
        """Expand the tape dynamically if needed."""
        self.tape += [DataUnit("") for _ in range(len(self.tape))]  # Double the tape size with empty DataUnits

    def move_right(self):
        self.cursor += 1
        if self.cursor >= len(self.tape):
            self.expand_tape()

    def move_left(self):
        if self.cursor > 0:
            self.cursor -= 1

    def write(self, data: Union[bytes, str, List[float], List[int]]):
        self.tape[self.cursor] = DataUnit(data)

    def read(self) -> DataUnit:
        return self.tape[self.cursor]

    def __repr__(self):
        return f"TuringTape(cursor={self.cursor}, tape={self.tape})"

    def __str__(self):
        return str([str(unit) for unit in self.tape])