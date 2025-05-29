import random
import time

class WillState:
    """
    A state representing the manifestation of 'Will' in a computational entity.
    Morphologically expressive. Thermodynamically costly. Philosophically recursive.
    """

    def __init__(self, energy=100, entropy=0):
        self.energy = energy     # Finite resource
        self.entropy = entropy   # Growing disorder
        self.thoughts = []       # Representations born from will
        self.depth = 0           # Recursive depth of cognition

    def __repr__(self):
        return f"<WillState | E:{self.energy} | S:{self.entropy:.2f} | T:{len(self.thoughts)} | D:{self.depth}>"

    def think(self):
        """Will generates thought-like patterns."""
        if self.energy <= 0:
            print("Will exhausted.")
            return False

        # Thought generation consumes energy and increases entropy
        effort = random.uniform(0.5, 3.0)
        self.energy -= effort
        self.entropy += effort * 0.2
        self.depth += 1

        # Thought is a symbolic representation of striving
        thought = {
            "pattern": random.choice(['loop', 'branch', 'repeat', 'halt']),
            "depth": self.depth,
            "cost": effort,
            "timestamp": len(self.thoughts)
        }
        self.thoughts.append(thought)

        return True

    def recurse(self):
        """Recursive expression of will"""
        while self.think():
            child_state = WillState(
                energy=self.energy / 2,
                entropy=self.entropy
            )
            child_outcome = child_state.run(limit=10)
            self.entropy += child_outcome["entropy"]
            self.energy = self.energy / 2
            self.thoughts.extend(child_outcome["thoughts"])

    def run(self, limit=100):
        """Run the will until exhaustion or limit reached."""
        count = 0
        while count < limit and self.think():
            count += 1
            time.sleep(0.01)  # Simulate computation delay

        return {
            "thoughts": self.thoughts,
            "entropy": self.entropy,
            "remaining_energy": max(0, self.energy),
            "striving_depth": self.depth
        }

# ————————————————————————————————————————————————
#  The Main Loop: Striving without End
# ————————————————————————————————————————————————

def main():
    """
    Entry point into the simulation of Will.
    Feel free to tamper with initial conditions.
    """
    print("Initializing WillMachine...")
    mind = WillState(energy=200)

    print(f"Initial State: {mind}")
    outcome = mind.run(limit=200)

    print("\nFinal State:")
    print(outcome)

    print("\nRecursive Expression:")
    deep_mind = WillState(energy=150)
    deep_mind.recurse()
    print(deep_mind.thoughts[-5:])  # Show last thoughts

if __name__ == "__main__":
    main()