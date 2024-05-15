from src.app import AtomDataclass, T, FormalTheory
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, TypeVar

T = TypeVar('T')

def main():
    atom1 = AtomDataclass(1)
    atom2 = AtomDataclass(2)
    atom3 = AtomDataclass(1)

    theory = FormalTheory[int]()
    
    # Display atoms
    print("Atoms:")
    for atom in [atom1, atom2, atom3]:
        print(repr(atom))
    
    # Formatting output with better readability
    print(f"Theory:\n{theory}")
    
    print("\nComparison result", theory.compare([atom1, atom2, atom3]), "\n")
    print("\nReflexivity result", theory.reflexivity(atom1.value), "\n")
    print("\nSymmetry result", all(theory.symmetry(atom1.value, atom.value) for atom in [atom1, atom2, atom3]), "\n")
    print("\nTransitivity result", theory.transitivity(atom1.value, atom3.value, atom1.value), "\n")
    print("\nTransparency result", theory.transparency(theory.if_else, atom1.value, atom1.value), "\n")
    print("\nCase base result", theory.case_base['a'](atom1.value, atom2.value), "\n")

if __name__ == "__main__":
    main()