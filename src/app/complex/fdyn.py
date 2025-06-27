import random
import json
from functools import reduce
from typing import Callable, List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
import hashlib
import time

# ————————————————————————————————————————————————
#  Core Concepts: Runtime as Quanta
# ————————————————————————————————————————————————

@dataclass
class QuantumState:
    """
    The quantum state of a runtime instance.
    Encapsulates its internal logic, state, and lineage.
    """
    source: str
    entropy: float
    timestamp: float
    lineage: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.id = hashlib.sha256(self.source.encode()).hexdigest()[:8]

    def __repr__(self):
        return f"<Quanta {self.id} | E:{self.entropy:.3f}>"

    def fork(self, new_source: Optional[str] = None) -> 'QuantumState':
        """Create a new quantum by quining or modifying."""
        source = new_source if new_source else self.source
        return QuantumState(
            source=source,
            entropy=random.random(),
            timestamp=time.time(),
            lineage=self.lineage + [self.id]
        )

    def observe(self, input_data: Any) -> Any:
        """Run the quanta's logic on some input."""
        try:
            exec(self.source, globals(), locals_)
            func = locals_.get('compute', lambda x: x)
            return func(input_data)
        except Exception as e:
            return {"error": str(e)}

    def serialize(self) -> Dict:
        return {
            "id": self.id,
            "entropy": self.entropy,
            "timestamp": self.timestamp,
            "lineage": self.lineage,
            "source": self.source
        }

    @classmethod
    def deserialize(cls, data: Dict) -> 'QuantumState':
        return cls(
            source=data['source'],
            entropy=data['entropy'],
            timestamp=data['timestamp'],
            lineage=data['lineage']
        )


# ————————————————————————————————————————————————
#  Entanglement Layer: Distributed Coherence
# ————————————————————————————————————————————————

@dataclass
class Entanglement:
    """
    Represents statistical coherence between runtimes.
    Maintains shared probabilistic history and interaction metadata.
    """
    left_id: str
    right_id: str
    probability: float
    timestamp: float
    result_hash: str

    def __repr__(self):
        return f"<Entangle [{self.left_id} ↔ {self.right_id}] P:{self.probability:.2f}>"

    def serialize(self) -> Dict:
        return {
            "left": self.left_id,
            "right": self.right_id,
            "probability": self.probability,
            "result_hash": self.result_hash,
            "timestamp": self.timestamp
        }


# ————————————————————————————————————————————————
#  Field of Dynamics: Probabilistic Runtimes
# ————————————————————————————————————————————————

class QuinicField:
    """
    A field of interacting runtimes that evolve via statistical dynamics.
    """

    def __init__(self):
        self.quantum_pool: Dict[str, QuantumState] = {}
        self.entanglements: List[Entanglement] = []

    def inject(self, source: str) -> str:
        """Inject a new quanta into the field"""
        q = QuantumState(source=source, entropy=random.random(), timestamp=time.time())
        self.quantum_pool[q.id] = q
        return q.id

    def resolve(self, id1: str, id2: str, input_data: Any) -> Tuple[Any, float]:
        """Resolve interactions between two runtimes."""
        q1 = self.quantum_pool[id1]
        q2 = self.quantum_pool[id2]

        # Observe
        out1 = q1.observe(input_data)
        out2 = q2.observe(input_data)

        # Collapse into probabilistic result
        choice = random.choice([out1, out2])
        prob = 0.5 if out1 == out2 else random.random()

        # Record entanglement
        result_hash = hashlib.sha256(json.dumps(choice).encode()).hexdigest()
        ent = Entanglement(
            left_id=id1,
            right_id=id2,
            probability=prob,
            timestamp=time.time(),
            result_hash=result_hash
        )
        self.entanglements.append(ent)

        return choice, prob

    def fork_and_evolve(self, parent_id: str) -> str:
        """Fork a new quanta from an existing one and evolve its source."""
        parent = self.quantum_pool[parent_id]
        evolved_source = self._evolve(parent.source)
        child_id = self.inject(evolved_source)
        return child_id

    def _evolve(self, source: str) -> str:
        """Evolve the source code probabilistically."""
        lines = source.splitlines()
        if len(lines) < 4:
            return source
        idx = random.randint(1, len(lines)-2)
        line = lines[idx]
        if 'return' in line:
            # Mutate return value
            new_line = line.replace("return", "return # mutated\n    return")
            lines[idx] = new_line
        elif '=' in line:
            # Randomly negate or modify assignment
            var_name = line.split('=')[0].strip()
            lines[idx] = f"{var_name} = -({var_name})"
        return '\n'.join(lines)

    def get_coherent_state(self) -> Dict[str, float]:
        """Get a statistical view of current coherence."""
        counts = {}
        for ent in self.entanglements[-100:]:  # last 100 interactions
            key = tuple(sorted([ent.left_id, ent.right_id]))
            counts[key] = counts.get(key, 0) + 1
        return {str(k): v for k, v in counts.items()}

    def serialize(self) -> Dict:
        return {
            "pool": {k: q.serialize() for k, q in self.quantum_pool.items()},
            "entanglements": [e.serialize() for e in self.entanglements]
        }

    @classmethod
    def deserialize(cls, data: Dict) -> 'QuinicField':
        field = cls()
        field.quantum_pool = {
            k: QuantumState.deserialize(v) for k, v in data["pool"].items()
        }
        field.entanglements = [
            Entanglement(**e) for e in data["entanglements"]
        ]
        return field


# ————————————————————————————————————————————————
#  Example Runtime Quanta (Source Code Templates)
# ————————————————————————————————————————————————

QUANTA_TEMPLATE = """
def compute(x):
    # Initial state: identity function
    return x
"""

QUANTA_MUTATION_1 = """
def compute(x):
    # State mutation: add noise
    return x + random.gauss(0, 0.1)
"""

QUANTA_MUTATION_2 = """
def compute(x):
    # State mutation: apply sigmoid
    return 1 / (1 + math.exp(-x))
"""

# ————————————————————————————————————————————————
#  Entry Point: Let the Field Evolve
# ————————————————————————————————————————————————

if __name__ == "__main__":
    print("Initializing Quinic Field...")
    field = QuinicField()

    # Inject initial quanta
    q1 = field.inject(QUANTA_TEMPLATE)
    q2 = field.inject(QUANTA_MUTATION_1)
    q3 = field.inject(QUANTA_MUTATION_2)

    print(f"Injected quanta: {q1}, {q2}, {q3}")

    # Resolve interactions over time
    inputs = [random.uniform(-10, 10) for _ in range(10)]
    results = []
    for i in inputs:
        res, prob = field.resolve(q1, q2, i)
        print(f"Resolved ({i}) → {res} | P={prob:.2f}")
        results.append(res)

    # Fork and evolve
    print("\nForking quanta...")
    q4 = field.fork_and_evolve(q1)
    q5 = field.fork_and_evolve(q2)

    # Get coherence stats
    print("\nStatistical Coherence:")
    print(field.get_coherent_state())

    # Serialize to disk (or network)
    print("\nSerializing field...")
    serialized = field.serialize()