#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
# © 2025 Moonlapsed https://github.com/MOONLAPSED/Cognosis | CC ND && BSD-3 | SEE LICENCE
import tkinter as tk
from tkinter import ttk
from typing import Tuple
from random import randint, seed as rand_seed
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
import random
import math

class MorphicRole(Enum):
    COMPUTE = 0  # C - operator/function aspect
    VALUE = 1    # V - state/value aspect  
    TYPE = 2     # T - type/structural aspect

@dataclass
class BitMorphology:
    raw: int  # Raw 8-bit value

    @property
    def bra(self) -> int:
        return (self.raw >> 4) & 0xF

    @property
    def ket(self) -> int:
        return self.raw & 0xF

    @property
    def compute(self) -> bool:
        return bool((self.raw >> 7) & 0x1)

    @property
    def value_triplet(self) -> Tuple[bool, bool, bool]:
        return (
            bool((self.raw >> 6) & 0x1),
            bool((self.raw >> 5) & 0x1),
            bool((self.raw >> 4) & 0x1)
        )

    @property
    def type_quartet(self) -> Tuple[bool, bool, bool, bool]:
        return (
            bool((self.raw >> 3) & 0x1),
            bool((self.raw >> 2) & 0x1),
            bool((self.raw >> 1) & 0x1),
            bool(self.raw & 0x1)
        )

    def morph(self, other: 'BitMorphology') -> 'BitMorphology':
        if self.compute:
            result = (self.bra << 4) | other.ket
        else:
            result = (other.bra << 4) | self.ket
        return BitMorphology(result)

    def inner_product(self, other: 'BitMorphology') -> int:
        bra_diff = self.bra ^ other.bra
        ket_diff = self.ket ^ other.ket
        return 8 - bin(bra_diff).count('1') - bin(ket_diff).count('1')

    def evolve(self) -> 'BitMorphology':
        if self.compute:
            new_raw = ((self.raw & 0x1) << 7) | (self.raw >> 1)
        else:
            new_raw = ((self.raw << 1) & 0xFF) | ((self.raw >> 7) & 0x1)
        return BitMorphology(new_raw)

    @staticmethod
    def create_from_components(
        compute: bool,
        values: Tuple[bool, bool, bool],
        types: Tuple[bool, bool, bool, bool]
    ) -> 'BitMorphology':
        raw = (int(compute) << 7)
        raw |= (int(values[0]) << 6)
        raw |= (int(values[1]) << 5)
        raw |= (int(values[2]) << 4)
        raw |= (int(types[0]) << 3)
        raw |= (int(types[1]) << 2)
        raw |= (int(types[2]) << 1)
        raw |= int(types[3])
        return BitMorphology(raw)

    def __repr__(self) -> str:
        c = "C" if self.compute else "c"
        v = "".join(["V" if v else "v" for v in self.value_triplet])
        t = "".join(["T" if t else "t" for t in self.type_quartet])
        return f"{c}{v}|{t} [0x{self.raw:02x}]"


class MorphicField:
    def __init__(self, size: Tuple[int, int]):
        self.width, self.height = size
        self.lattice = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.morphologies = [[BitMorphology(0) for _ in range(self.width)] for _ in range(self.height)]

    def initialize_random(self, seed: Optional[int] = None) -> None:
        rng = random.Random(seed)
        for y in range(self.height):
            for x in range(self.width):
                val = rng.randint(0, 255)
                self.lattice[y][x] = val
                self.morphologies[y][x] = BitMorphology(val)

    def initialize_kernel(self, kernel_type: str) -> None:
        if kernel_type == "gaussian_white":
            rng = random.Random()
            for y in range(self.height):
                for x in range(self.width):
                    val = int(rng.gauss(128, 40))
                    val = max(0, min(255, val))
                    self.lattice[y][x] = val
                    self.morphologies[y][x] = BitMorphology(val)

        elif kernel_type == "quine":
            midpoint = (self.width // 2, self.height // 2)
            for y in range(self.height):
                for x in range(self.width):
                    dx = x - midpoint[0]
                    dy = y - midpoint[1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    compute = (int(dist) % 2 == 0)
                    values = (dist % 3 < 1, dist % 5 < 2, dist % 7 < 3)
                    types = (dist % 2 < 1, dist % 3 < 1, dist % 5 < 2, dist % 7 < 3)
                    morph = BitMorphology.create_from_components(compute, values, types)
                    self.lattice[y][x] = morph.raw
                    self.morphologies[y][x] = morph

    def step(self) -> None:
        new_lattice = [[0 for _ in range(self.width)] for _ in range(self.height)]
        new_morphologies = [[BitMorphology(0) for _ in range(self.width)] for _ in range(self.height)]

        for y in range(self.height):
            for x in range(self.width):
                center = self.morphologies[y][x]
                neighbors = self._get_neighbors(x, y)
                new_morph = self._apply_rulial_dynamics(center, neighbors)
                new_lattice[y][x] = new_morph.raw
                new_morphologies[y][x] = new_morph

        self.lattice = new_lattice
        self.morphologies = new_morphologies

    def _get_neighbors(self, x: int, y: int) -> List[BitMorphology]:
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx = (x + dx) % self.width
                ny = (y + dy) % self.height
                neighbors.append(self.morphologies[ny][nx])
        return neighbors

    def _apply_rulial_dynamics(self, center: BitMorphology, neighbors: List[BitMorphology]) -> BitMorphology:
        max_resonance = -1
        most_resonant = None
        for neighbor in neighbors:
            resonance = center.inner_product(neighbor)
            if resonance > max_resonance:
                max_resonance = resonance
                most_resonant = neighbor

        if max_resonance >= 6:
            return center.morph(most_resonant)
        elif max_resonance >= 4:
            return center.evolve()
        else:
            return BitMorphology(center.raw)

    def visualize(self) -> List[List[Tuple[int, int, int]]]:
        """
        Return a 2D array of RGB tuples representing the field state.
        """
        image = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                morph = self.morphologies[y][x]
                r = 255 if morph.compute else 0
                g = int(sum(morph.value_triplet) * 255 / 3)
                b = int(sum(morph.type_quartet) * 255 / 4)
                row.append((r, g, b))
            image.append(row)
        return image

class MorphicApp:
    def __init__(self, root: tk.Tk, field_size: Tuple[int, int] = (64, 64)):
        self.root = root
        self.root.title("Morphic Field Visualizer")
        
        self.width, self.height = field_size
        self.field = MorphicField((self.width, self.height))
        self.field.initialize_random(seed=42)

        self.canvas = tk.Canvas(root, width=self.width, height=self.height)
        self.canvas.pack()

        # Control buttons
        control_frame = ttk.Frame(root)
        control_frame.pack(pady=5)
        ttk.Button(control_frame, text="Step", command=self.step).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Reset Random", command=self.reset_random).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Quine Kernel", command=self.reset_quine).pack(side=tk.LEFT)

        self.image = tk.PhotoImage(width=self.width, height=self.height)
        self.canvas.create_image((0, 0), image=self.image, anchor="nw")

        self.update_display()

    def step(self):
        self.field.step()
        self.update_display()

    def reset_random(self):
        self.field.initialize_random()
        self.update_display()

    def reset_quine(self):
        self.field.initialize_kernel("quine")
        self.update_display()

    def update_display(self):
        for y in range(self.height):
            for x in range(self.width):
                morph = self.field.morphologies[y][x]
                r = 255 if morph.compute else 0
                g = int(sum(morph.value_triplet) * 255 / 3)
                b = int(sum(morph.type_quartet) * 255 / 4)
                hex_color = f'#{r:02x}{g:02x}{b:02x}'
                self.image.put(hex_color, (x, y))

def main():
    root = tk.Tk()
    app = MorphicApp(root, field_size=(64, 64))
    root.mainloop()

if __name__ == "__main__":
    main()
