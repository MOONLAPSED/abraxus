#!/usr/bin/env python3
"""
Morphological Query System for evolving knowledge representations.
This system enables quantum-inspired transformations of knowledge queries.
"""

import re
import json
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import time

# Reference to KnowledgeBase from previous artifact
from .morphofs import KnowledgeBase, KnowledgeEntry, __Atom__, FrameModel

@dataclass
class QueryVector:
    """
    Represents a query as a vector in semantic space.
    Similar to your concept of state vectors in Hilbert space.
    """
    tokens: List[str]
    weights: np.ndarray
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if len(self.tokens) != len(self.weights):
            raise ValueError("Tokens and weights must have the same length")
    
    def normalize(self) -> 'QueryVector':
        """Normalize the weight vector (similar to quantum state normalization)."""
        norm = np.linalg.norm(self.weights)
        if norm > 0:
            normalized_weights = self.weights / norm
            return QueryVector(self.tokens, normalized_weights, self.context)
        return self
    
    def superposition(self, other: 'QueryVector', alpha: float = 0.5) -> 'QueryVector':
        """
        Create a superposition of two query vectors.
        Similar to quantum superposition where states can exist simultaneously.
        """
        # Find common tokens and unique tokens
        all_tokens = list(set(self.tokens + other.tokens))
        
        # Create new weight vector with proper sizes
        new_weights = np.zeros(len(all_tokens))
        
        # Transfer weights from both vectors
        for i, token in enumerate(all_tokens):
            if token in self.tokens:
                idx = self.tokens.index(token)
                new_weights[i] += alpha * self.weights[idx]
            
            if token in other.tokens:
                idx = other.tokens.index(token)
                new_weights[i] += (1 - alpha) * other.weights[idx]
        
        # Merge contexts
        merged_context = {}
        if self.context:
            merged_context.update(self.context)
        if other.context:
            merged_context.update(other.context)
        
        return QueryVector(all_tokens, new_weights, merged_context).normalize()


class MorphologicalOperator:
    """
    Represents a transformation operator that can be applied to query vectors.
    Similar to your concept of self-adjoint operators in Hilbert space.
    """
    def __init__(self, name: str, transform_fn: Callable[[QueryVector], QueryVector]):
        self.name = name
        self.transform = transform_fn
    
    def apply(self, query_vector: QueryVector) -> QueryVector:
        """Apply the operator transformation to a query vector."""
        return self.transform(query_vector)
    
    def is_self_adjoint(self, test_vectors: List[QueryVector], epsilon: float = 1e-6) -> bool:
        """
        Test if the operator is approximately self-adjoint on the given set of vectors.
        Self-adjoint operators are essential in quantum mechanics and in your theory.
        """
        for v in test_vectors:
            # Apply the operator
            transformed = self.apply(v)
            
            # For a self-adjoint operator A, we should have <v|A|v> = <Av|v>
            # In our simplified case, we'll check if the dot product is real
            # and if applying twice gives back a scaled version of the original
            twice_transformed = self.apply(transformed)
            
            # Check if the twice-transformed vector is parallel to the original
            if len(v.tokens) != len(twice_transformed.tokens):
                return False
                
            # Simplistic check - this would be more complex with proper Hilbert space
            original_norm = np.linalg.norm(v.weights)
            transformed_norm = np.linalg.norm(transformed.weights)
            twice_transformed_norm = np.linalg.norm(twice_transformed.weights)
            
            if abs(twice_transformed_norm - original_norm) > epsilon:
                return False
        
        return True


class MorphologicalQueryEngine:
    """
    Engine for executing and evolving morphological queries over a knowledge base.
    Implements the quantum-inspired computation model for knowledge retrieval.
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.operators: Dict[str, MorphologicalOperator] = {}
        self.register_default_operators()
        self.query_history: List[Tuple[QueryVector, List[str]]] = []
    
    def register_default_operators(self):
        """Register the default set of operators."""
        
        # Expansion operator - broadens a query by reducing weights of specific terms
        def expand_transform(qv: QueryVector) -> QueryVector:
            new_weights = qv.weights * 0.8  # Reduce specificity
            return QueryVector(qv.tokens, new_weights, qv.context).normalize()
        
        # Focus operator - narrows a query by amplifying the highest weighted terms
        def focus_transform(qv: QueryVector) -> QueryVector:
            # Find the max weight
            max_weight = np.max(qv.weights)
            # Amplify weights close to max, suppress others
            new_weights = np.where(qv.weights > 0.7 * max_weight, 
                                   qv.weights * 1.3, 
                                   qv.weights * 0.5)
            return QueryVector(qv.tokens, new_weights, qv.context).normalize()
        
        # Orthogonal operator - emphasizes terms not strongly present
        def orthogonal_transform(qv: QueryVector) -> QueryVector:
            # Invert the weights (with some scaling to avoid extremes)
            max_weight = np.max(qv.weights)
            if max_weight > 0:
                new_weights = max_weight - qv.weights
                return QueryVector(qv.tokens, new_weights, qv.context).normalize()
            return qv
        
        # Register the operators
        self.operators["expand"] = MorphologicalOperator("expand", expand_transform)
        self.operators["focus"] = MorphologicalOperator("focus", focus_transform)
        self.operators["orthogonal"] = MorphologicalOperator("orthogonal", orthogonal_transform)
    
    def text_to_query_vector(self, text: str) -> QueryVector:
        """Convert a text query into a query vector."""
        # Simple tokenization (in practice, use a better NLP approach)
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Remove duplicates while preserving order
        unique_tokens = []
        for token in tokens:
            if token not in unique_tokens:
                unique_tokens.append(token)
        
        # Initialize weights (uniform for now)
        weights = np.ones(len(unique_tokens))
        
        return QueryVector(unique_tokens, weights).normalize()
    
    def apply_operator(self, query_vector: QueryVector, operator_name: str) -> QueryVector:
        """Apply a named operator to a query vector."""
        if operator_name not in self.operators:
            raise ValueError(f"Unknown operator: {operator_name}")
        
        return self.operators[operator_name].apply(query_vector)
    
    def execute_query(self, query_vector: QueryVector) -> List[KnowledgeEntry]:
        """
        Execute a query vector against the knowledge base.
        This is where the "measurement" of your quantum-inspired system happens.
        """
        # Convert query vector to a weighted text search
        query_terms = [(token, weight) for token, weight in zip(query_vector.tokens, query_vector.weights)]
        
        # Sort by weight to prioritize the most important terms
        query_terms.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        seen_ids = set()
        
        # Start with highest weighted terms
        for term, weight in query_terms:
            if weight > 0.2:  # Only use terms with significant weight
                # Search the knowledge base
                term_results = self.kb.search(term)
                
                for entry in term_results:
                    if entry.id not in seen_ids:
                        results.append(entry)
                        seen_ids.add(entry.id)
        
        # Store query history for later analysis
        self.query_history.append((query_vector, list(seen_ids)))
        
        return results
    
    def quantum_search(self, text_query: str, operators: List[str] = None) -> List[KnowledgeEntry]:
        """
        Perform a quantum-inspired search that applies operators to transform the query.
        This implements the core of your morphological source code concept.
        """
        # Convert text to initial query vector
        qv = self.text_to_query_vector(text_query)
        
        # Apply operators in sequence if specified
        if operators:
            for op_name in operators:
                qv = self.apply_operator(qv, op_name)
        
        # Execute the transformed query
        return self.execute_query(qv)
    
    def evolve_query(self, initial_query: str, iterations: int = 3) -> List[KnowledgeEntry]:
        """
        Evolve a query over multiple iterations, learning from results.
        This implements the agentic motility concept from your theory.
        """
        qv = self.text_to_query_vector(initial_query)
        
        all_results = []
        seen_ids = set()
        
        for i in range(iterations):
            # Apply a random operator to introduce quantum-like uncertainty
            op_name = np.random.choice(list(self.operators.keys()))
            qv = self.apply_operator(qv, op_name)
            
            # Execute the transformed query
            iteration_results = self.execute_query(qv)
            
            # Analyze results to adjust the query vector
            if iteration_results:
                # Extract keywords from results to enhance the query
                new_tokens = []
                for entry in iteration_results:
                    new_tokens.extend(entry.keywords)
                
                # Filter to unique tokens not already in query
                new_tokens = [t for t in new_tokens if t not in qv.tokens]
                
                if new_tokens:
                    # Create a new query vector from the keywords
                    keyword_weights = np.ones(len(new_tokens)) * 0.5
                    keyword_qv = QueryVector(new_tokens, keyword_weights)
                    
                    # Create a superposition of the original and new vectors
                    qv = qv.superposition(keyword_qv, 0.7)
            
            # Collect unique results
            for entry in iteration_results:
                if entry.id not in seen_ids:
                    all_results.append(entry)
                    seen_ids.add(entry.id)
            
            # Small delay to simulate quantum decoherence
            time.sleep(0.1)
        
        return all_results
    
    def get_query_history_graph(self) -> Dict[str, Any]:
        """
        Generate a graph representation of the query evolution history.
        This helps visualize how queries transform over time.
        """
        nodes = []
        edges = []
        
        # Add nodes for each query
        for i, (query_vector, result_ids) in enumerate(self.query_history):
            # Create a representative query string
            query_str = " ".join([f"{t}:{w:.2f}" for t, w in 
                                 zip(query_vector.tokens, query_vector.weights) 
                                 if w > 0.2])
            
            nodes.append({
                "id": f"q{i}",
                "label": query_str,
                "type": "query",
                "results": len(result_ids)
            })
            
            # Connect to previous query if exists
            if i > 0:
                edges.append({
                    "source": f"q{i-1}",
                    "target": f"q{i}",
                    "label": "evolves_to"
                })
            
            # Connect to results
            for result_id in result_ids:
                # Add result node if not already added
                result_node_id = f"r{result_id}"
                if not any(n["id"] == result_node_id for n in nodes):
                    entry = self.kb.get_entry(result_id)
                    if entry:
                        nodes.append({
                            "id": result_node_id,
                            "label": entry.title,
                            "type": "result"
                        })
                
                edges.append({
                    "source": f"q{i}",
                    "target": result_node_id,
                    "label": "returns"
                })
        
        return {
            "nodes": nodes,
            "edges": edges
        }


def main():
    """Demo of the morphological query system."""
    kb = KnowledgeBase()
    
    # Add some sample entries (reusing previous examples)
    quantum_id = kb.add_entry("""---
name: "Quantum Computing Basics"
tags: ["quantum", "computing", "theory"]
keywords: ["qubit", "superposition", "entanglement"]
---
# Quantum Computing Basics

Quantum computing uses quantum-mechanical phenomena to perform computation.
Unlike classical computers, quantum computers use qubits which can represent
a superposition of states.

See also: [[Hilbert Space]], [[Quantum Gates]]
""")
    
    hilbert_id = kb.add_entry("""---
name: "Hilbert Space"
tags: ["quantum", "theory"]
keywords: ["vector space", "inner product", "density matrix"]
---
# Hilbert Space

Hilbert space is a mathematical framework that generalizes the concept of a vector space.
It is used to describe quantum states and their interactions.

See also: [[Quantum Computing Basics]], [[Quantum Gates]]
""")

    gates_id = kb.add_entry("""---
name: "Quantum Gates"
tags: ["quantum", "computing", "theory"]
keywords: ["unitary", "matrix", "quantum logic gate"]
---
""")

    # Initialize the query engine
    engine = MorphologicalQueryEngine(kb)

    # Perform a quantum-inspired search
    results = engine.quantum_search("quantum computing", operators=["expand", "focus"])
    for entry in results:
        print(entry.title)

    # Evolve a query over multiple iterations
    evolved_results = engine.evolve_query("quantum computing", iterations=5)
    for entry in evolved_results:
        print(entry.title)

    # Visualize the query history
    graph = engine.get_query_history_graph()
    print(json.dumps(graph, indent=4))


if __name__ == "__main__":
    main()
