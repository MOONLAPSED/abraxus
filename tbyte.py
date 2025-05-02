import tkinter as tk
from tkinter import ttk
from random import randint
from enum import Enum

class ByteWord:
    def __init__(self, value=0):
        self._value = value & 0xFF

    @property
    def value(self): return self._value
    @value.setter
    def value(self, v): self._value = v & 0xFF

    @property
    def high_nibble(self): return (self._value >> 4) & 0x0F
    @property
    def morphism_selector(self): return (self._value >> 1) & 0x07
    @property
    def control_bit(self): return self._value & 0x01
    def __repr__(self): return f"{self._value:02X}"

class AddressingMode(Enum):
    DIRECT = 0
    RECURSIVE = 1

class HolographicMemory:
    def __init__(self, mode=AddressingMode.RECURSIVE):
        self.memory = {i: ByteWord(randint(0, 255)) for i in range(16)}

    def get_state(self): return self.memory

    def evolve(self):
        """Dummy evolve function: Flip control bits randomly for demo."""
        for bw in self.memory.values():
            if randint(0, 1):
                bw.value ^= 0x01  # Flip C-bit (simulate energy decay)

# --- Tkinter Visualizer ---

class MemoryVisualizer(tk.Tk):
    def __init__(self, memory: HolographicMemory):
        super().__init__()
        self.title("ByteWord Memory Visualizer")
        self.geometry("400x300")
        self.memory = memory
        self.cells = {}

        self.grid_frame = ttk.Frame(self)
        self.grid_frame.pack(padx=10, pady=10)

        self._build_grid()
        self._draw_memory()

        self.step_button = ttk.Button(self, text="Step", command=self.step)
        self.step_button.pack(pady=5)

    def _build_grid(self):
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                label = tk.Label(
                    self.grid_frame,
                    text="",
                    width=8,
                    height=2,
                    relief="raised",
                    bg="gray"
                )
                label.grid(row=i, column=j, padx=4, pady=4)
                self.cells[idx] = label

    def _draw_memory(self):
        for addr, bw in self.memory.get_state().items():
            label = self.cells[addr]
            label.config(
                text=f"{addr:02d}\n{bw}",
                bg="green" if bw.control_bit else "black",
                fg="white"
            )

    def step(self):
        self.memory.evolve()
        self._draw_memory()

# --- Run the Demonstrator ---

if __name__ == "__main__":
    mem = HolographicMemory()
    app = MemoryVisualizer(mem)
    app.mainloop()