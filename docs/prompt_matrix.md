# 3x3x3 Scientific Prompt Matrix

This document describes the structured dataset used for the idempotency experiments. To ensure a broad and representative study of convergence behavior, we use a **3x3x3 Scientific Matrix** consisting of 27 distinct prompts.

The matrix is organized along two primary dimensions: **Complexity** and **Domain**.

## Dimensions

### 1. Complexity (The Structural Type)
*   **Simple (Linear)**: Straight-forward, one-way flows with no feedback loops or deep nesting. Measures basic symbol preservation.
*   **Cyclic (Feedback)**: Systems containing closed loops or iterative processes. Measures how models handle recurrence and self-correction.
*   **Hierarchical (Nested)**: Systems with parent-child relationships or multiple levels of abstraction. Measures structural depth preservation.

### 2. Domain (The Conceptual Area)
*   **Software**: Technical, logic-heavy systems (e.g., Auth flows, Microservices).
*   **Physical**: Real-world mechanical or biological systems (e.g., Engine cycles, Assembly lines).
*   **Abstract**: Theoretical, logical, or narrative structures (e.g., Syllogisms, Hero's Journey).

---

## The Matrix (27 Prompts)

| Complexity \ Domain | Software | Physical | Abstract |
| :--- | :--- | :--- | :--- |
| **Simple** | User Auth, Password Reset, Email Sub | Water Filter, Assembly Line, Traffic Lights | Syllogism, Event Timeline, Math Derivation |
| **Cyclic** | CI/CD Pipeline, Event Sourcing, Garbage Collection | Thermostat, Water Cycle, Engine Cycle | Hero's Journey, Scientific Method, Design Thinking |
| **Hierarchical** | Microservices, Kubernetes, React Component Tree | Solar System, Trophic Pyramid, Nervous System | Taxonomy, Corporate Org Chart, Decision Tree |

---

## Rationale
By crossing these dimensions, we can isolate whether convergence properties are:
1.  **Complexity-dependent**: Do cyclic systems "descend" into simpler linear systems over time?
2.  **Domain-dependent**: Does the "Physical" grounding help models maintain stability compared to "Abstract" logic?
3.  **Universally Attractor-driven**: Do all categories eventually reach a similar state of "semantic minimum" regardless of their starting point?

This matrix is implemented in `config/prompts/batch_stage2_robust.txt`.
