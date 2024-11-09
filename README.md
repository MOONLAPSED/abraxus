![welcome](/kb/assets/pub/image.jpg)

# Abraxus

## Usage; testing, fuzzing, linting, etc.

```bash
make install
make lint
make format
make test
make bench
make pre-commit-install
pdm run python main.py
```

## first run in a minit?

`python -m pip install -U pdm`

OR, if a refreshing anew:

```
pipx upgrade pdm
pdm update
pdm cache clear
pdm install --clean
```

### usage/install context

See /docs/install.md for preparing a machine or for step-by-step instructions if the above is not self explanitory

1) Install windows, update to service pack >= 23H2 - do not 'connect automatically' to wifi (checkbox)... (see docs for more)


## what is 'motility' & 'CCC'?

[[Agentic Motility System]]

**Overview:**
The Agentic Motility System is an architectural paradigm for creating AI agents that can dynamically extend and reshape their own capabilities through a cognitively coherent cycle of reasoning and source code evolution.

**Key Components:**
- **Hard Logic Source (db)**: The ground truth implementation that instantiates the agent's initial logic and capabilities as hard-coded source.
- **Soft Logic Reasoning**: At runtime, the agent can interpret and manipulate the hard logic source into a flexible "soft logic" representation to explore, hypothesize, and reason over.
- **Cognitive Coherence Co-Routines**: Processes that facilitate shared understanding between the human and the agent to responsibly guide the agent's soft logic extrapolations.
- **Morphological Source Updates**: The agent's ability to propose modifications to its soft logic representation that can be committed back into the hard logic source through a controlled pipeline.
- **Versioned Runtime (kb)**: The updated hard logic source instantiates a new version of the agent's runtime, allowing it to internalize and build upon its previous self-modifications.

**The Motility Cycle:**
1. Agent is instantiated from a hard logic source (db) into a runtime (kb) 
2. Agent translates hard logic into soft logic for flexible reasoning
3. Through cognitive coherence co-routines with the human, the agent refines and extends its soft logic
4. Agent proposes soft logic updates to go through a pipeline to generate a new hard logic source 
5. New source instantiates an updated runtime (kb) for a new agent/human to build upon further

By completing and iterating this cycle, the agent can progressively expand its own capabilities through a form of "morphological source code" evolution, guided by its coherent collaboration with the human developer.

**Applications and Vision:**
This paradigm aims to create AI agents that can not only learn and reason, but actively grow and extend their own core capabilities over time in a controlled, coherent, and human-guided manner. Potential applications span domains like open-ended learning systems, autonomous software design, decision support, and even aspects of artificial general intelligence (AGI).

**training, RLHF, outcomes, etc.**
Every CCC db is itself a type of training and context but built specifically for RUNTIME abstract agents and specifically not for concrete model training. This means that you can train a CCC db with a human, but you can also train a CCC db with a RLHF agent. This is a key distinction between CCC and RLHF. In other words, every CCCDB is like a 'model' or an 'architecture' for a RLHF agent to preform runtime behavior within such that the model/runtime itself can enable agentic motility - with any LLM 'model' specifically designed for consumer usecases and 'small' large language models.







## Best practices:
- Utilize camelCase for internal source code
- Utilize CAPS_CASE for ffi funcs and external source


"""
## Frontmatter Implementation

 - Utilize 'frontmatter' to include the title and other `property`, `tag`, etc. in the knowledge base article(s).
   
   - For Example:
      ```
      ---
      name: "Article Title"
      link: "[[Related Link]]"
      linklist:
        - "[[Link1]]"
        - "[[Link2]]"
      ---
      ``` """




Core Ideas:
    Interactive Runtime Environments: You're contemplating systems where both player behaviors and agent decisions inform and restructure each other, forming emergent, adaptive ecosystems.
    Bi-directional Learning: This reciprocal relationship fosters a deeper integration of human-like adaptability in AI systems, merging deterministic and statistical learning methodologies.

Dynamic Execution:
    Nonlinear Dynamics of Play and Inference: Players navigate and modify their environment actively, while ML agents iterate on decisions, learning in real-time.
    Anticipatory Computation: Both paradigms involve predicting future states, aligning with anticipatory systems that adjust based on potential future configurations rather than solely historical data.

Innovations and Applications:
    Morphological Source Code: This concept involves source code that evolves with system state, expanding possibilities for self-modifying code that can dynamically represent and transform application behavior.
    Live Feedback and Adaptability: Techniques from live coding and agile development can inform AI model training, making real-time state management inherent to AI systems.
    Cross-Domain Fusion: By integrating gaming techniques (like game-state interaction) with machine learning, you could develop systems where AI and interactive environments inform each other symbiotically.



Zeroth Law (Holographic Foundation):
    Symbols and observations are perceived as real due to intrinsic system properties, creating self-consistent realities.

First and Second Laws:
    Adapt thermodynamic principles to computational contexts, allowing for self-regulation and prediction through the Free Energy Principle.

Binary Fundamentals and Complex Triads:
    0 and 1 are not just data but core "holoicons," representing more than bits—they are conceptual seeds from which entire computational universes can be constructed.
    The triadic approach (energy-state-logic) emphasizes a holistic computation model that blends deterministic systems with emergent phenomena.

Axiom of Potentiality and Observation:
    The system's state space includes all potential states, ontologically relevant only at the point of observation.
    This aligns with quantum concepts where potentialities collapse upon measurement.

Axiom of Morphogenesis and Recursion:
    Change and evolution are driven recursively, akin to self-modifying systems that adapt and self-repair—think of quines or self-referential algorithms.

Axiom of Holographic Compression:
    Reflects computational efficiency where each state can ideally be represented minimally without loss of detail, similar to practices in data compression and holographic information theory.

Axioms of Duality, Invariance, and Closure:
    These describe the system's intrinsic stability, ensuring transformations and interactions result in self-consistent layers of logic, retaining coherence even as complexity emerges.