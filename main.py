from src.app import AtomDataclass, FormalTheory

def main():
    atom1 = AtomDataclass(1, 1)
    atom2 = AtomDataclass(2, 2)
    atom3 = AtomDataclass(1, 1)

    theory = FormalTheory[int]()
    
    # Formatting output with better readability
    print(f"Theory:\n{theory}")
    print("\nComparison result:", theory.compare([atom1, atom2, atom3]), "\n")
    
    # Display atoms
    print("Atoms:")
    for atom in [atom1, atom2, atom3]:
        print(repr(atom))

if __name__ == "__main__":
    main()