# 🧬 SynB Complete Project Explanation Guide

Welcome to the **SynB Metabolic Engineering & Genome Reconstruction Platform** master guide. This document explains the scientific concepts, mathematical formulations, software architecture, and codebase details of the platform.

---

## 📚 1. Scientific & Biological Background

### Genome-Scale Metabolic Models (GEMs)
In synthetic biology, a cell is viewed as a **digital factory**. The DNA provides the instruction manuals (genes), which produce machinery (enzymes). These enzymes catalyze chemical reactions that convert raw feedstocks (metabolites like Glucose) into energy, cell wall components (biomass), and product outputs (like Ethanol).

A **Genome-Scale Metabolic Model (GEM)** is a structured database containing:
*   **Stoichiometric Matrix ($S$):** A mathematical representation of all biochemical reactions in the cell. If a model has $M$ metabolites and $R$ reactions, $S$ is an $M \times R$ matrix where $S_{ij}$ represents the stoichiometric coefficient of metabolite $i$ in reaction $j$.
*   **Gene-Protein-Reaction (GPR) Rules:** Boolean logic associations (e.g., `(GeneA AND GeneB) OR GeneC`) defining which enzymes catalyze which reactions.
*   **Bounds ($lb, ub$):** Lower and upper limits of reaction speeds (fluxes), representing thermodynamic constraints (e.g., irreversible reactions have $lb=0$) and nutrient availability limits (e.g., carbon source uptake rates).

### Constraint-Based Reconstruction & Analysis (COBRA)
Because measuring every single reaction speed in a living cell is impossible, we use **COBRA** methods. COBRA assumes the cell is in a **steady state**, meaning the concentration of internal metabolites remains constant over time (accumulation rate is zero):
$$\frac{dx}{dt} = S v = 0$$
where $v$ is the vector of all reaction fluxes. This forms a set of linear equations that constrains the feasible fluxes the cell can carry.

---

## 📐 2. Mathematical Formulations

SynB performs three primary types of optimization algorithms on metabolic models:

### 1. Flux Balance Analysis (FBA)
FBA solves a linear programming (LP) problem to find the optimal flux distribution that maximizes a biological objective (usually biomass/growth rate):

$$\begin{aligned}
\text{Maximize} \quad & c^T v \\
\text{subject to} \quad & S v = 0 \\
& lb_i \le v_i \le ub_i, \quad \forall i \in \{1, \dots, R\}
\end{aligned}$$

where:
*   $c$ is a vector of weights selecting the objective reaction (e.g., $c_i = 1$ for the biomass reaction, $0$ otherwise).
*   $v$ is the flux vector of the metabolic reactions.

---

### 2. Parsimonious FBA (pFBA)
Standard FBA can yield multiple mathematically equivalent optimal flux distributions, some of which may contain biologically unrealistic internal loops. pFBA resolves this by minimizing total enzymatic investment. It runs in two sequential LP steps:

1.  **Step 1:** Maximize growth rate to find the optimum value $z^* = \max c^T v$.
2.  **Step 2:** Minimize the sum of all absolute fluxes (L1-norm) while keeping growth rate at $z^*$:

$$\begin{aligned}
\text{Minimize} \quad & \sum_{i=1}^{R} |v_i| \\
\text{subject to} \quad & S v = 0 \\
& c^T v \ge z^* \\
& lb_i \le v_i \le ub_i, \quad \forall i \in \{1, \dots, R\}
\end{aligned}$$

This isolates the most efficient, parsimonious pathway mapping.

---

### 3. Flux Variability Analysis (FVA)
FVA evaluates the flexibility of the network. It calculates the minimum and maximum possible flux for *each individual reaction* that still satisfies the optimal growth requirement:

$$\begin{aligned}
\text{Minimize / Maximize} \quad & v_k \\
\text{subject to} \quad & S v = 0 \\
& c^T v \ge \alpha \cdot z^* \\
& lb_i \le v_i \le ub_i, \quad \forall i \in \{1, \dots, R\}
\end{aligned}$$

where $\alpha \in [0, 1]$ represents the fraction of optimal growth (e.g., $\alpha = 0.90$ for 90% growth).

---

### 4. OptKnock (Greedy LP Heuristic)
OptKnock identifies reaction knockouts that couple product synthesis with cell growth. While the rigorous formulation is a bilevel Mixed-Integer Linear Program (MILP), SynB uses a **greedy sequential search heuristic** to evaluate reaction deletions:

1.  Calculate max growth rate ($z^*$) under default conditions.
2.  Iterate through all non-essential reactions. Temporarily set bounds to $0$ ($lb_i = ub_i = 0$) for the candidate reaction.
3.  Solve the coupled LP to maximize product yield while forcing biomass production to be at least a fraction of the optimum:
    $$\max v_{\text{product}} \quad \text{s.t.} \quad S v = 0, \quad v_{\text{biomass}} \ge \alpha \cdot z^*$$
4.  Rank the knockouts, permanently disable the highest-ranking candidate, and repeat for the next knockout step.

---

## 🏗️ 3. Software & System Architecture

SynB is divided into three tiers:

```
┌────────────────────────────────────────────────────────┐
│               Streamlit Presentation Tier              │
│                     (app.py)                           │
└──────────────────────────┬─────────────────────────────┘
                           │ JSON/Multipart Form REST Requests
┌──────────────────────────▼─────────────────────────────┐
│                 FastAPI Gateway API Tier               │
│                  (backend/main.py)                     │
└──────────────────────────┬─────────────────────────────┘
                           │ Sandboxed Worker Threads
┌──────────────────────────▼─────────────────────────────┐
│                 Cobrapy Scientific Engine              │
│                 (backend/services/*)                   │
└────────────────────────────────────────────────────────┘
```

### 🧵 Thread Safety & Concurrent Solves
*   **The Problem:** COBRApy's underlying solver objects are written in C/C++ and hold mutable state. If two users concurrently call FBA on the same registered model, their solver settings (like bounds and tolerances) would collide and corrupt the results.
*   **The Solution:** 
    1.  **Isolation Context (`with model:`):** Every analysis function runs within Python's `with model:` context manager. This triggers COBRApy's transaction history stack, copying and isolating modifications (such as setting temporary bounds and solvers) and rolling them back completely once the execution completes.
    2.  **ThreadPoolExecutor:** All solver runs are offloaded from FastAPI's main event loop to a dedicated thread pool (`max_workers=4` inside [`backend/api/routes.py`](file:///Users/muhammedmidlaj/Desktop/arqgene-sybbio/backend/api/routes.py)). This prevents CPU-heavy optimizations from freezing the API.

### 🗑️ Memory Registry & Eviction (TTL)
Because metabolic models are loaded in memory, idle models must be cleaned up to prevent memory leaks. The system implements a **Time-To-Live (TTL)** eviction policy:
*   Models are kept in [`backend/services/model_registry.py`](file:///Users/muhammedmidlaj/Desktop/arqgene-sybbio/backend/services/model_registry.py) in a thread-safe dict.
*   An eviction loop runs in the background of FastAPI's lifespan loop every 10 minutes. If a model has not been read or modified for more than 1 hour (3600 seconds), it is automatically deleted.

---

## 🔬 4. Detailed Core Features & Pipelines

### 🛡️ 1. The 5-Point Validation Suite
Implemented in [`core/validation.py`](file:///Users/muhammedmidlaj/Desktop/arqgene-sybbio/core/validation.py), the quality validation engine performs:
1.  **Feasibility Check:** Checks if the objective function can carry positive flux. If growth rate $= 0$, it warns the user.
2.  **Inconsistent Bounds Check:** Scans for structural anomalies where lower bounds are greater than upper bounds ($lb > ub$).
3.  **Gene-Orphan Census:** Finds internal metabolic reactions lacking gene associations (GPR), exposing gaps in the genome annotation.
4.  **Blocked Reactions (via FVA):** Identifies reactions that are structurally constrained to carry $0$ flux under any environmental condition.
5.  **Mass Balance Auditing:** Validates that each reaction is elementally balanced (e.g., the carbon, nitrogen, hydrogen atoms on the left match the right). Unbalanced reactions indicate errors in the stoichiometry.

### 🧬 2. Genome-to-Model Reconstruction Pipeline
Located in [`core/genome_pipeline.py`](file:///Users/muhammedmidlaj/Desktop/arqgene-sybbio/core/genome_pipeline.py), this converts raw genomic FASTA sequences into simulation-ready models:
*   **Step 1 — Homology Mapping:** Runs sequence searches using the **DIAMOND** alignment tool against a curated database of enzymes ([`data/reference/reference_db.dmnd`](file:///Users/muhammedmidlaj/Desktop/arqgene-sybbio/data/reference/reference_db.dmnd)) to assign EC numbers.
*   **Step 2 — API Enrichment:** Queries the **KEGG REST API** to fetch metabolic reaction equations mapped to the detected EC numbers, with fallback queries to **UniProt**.
*   **Step 3 — Gap-Filling Heuristic:** Solves a mixed-integer optimization program to select the minimal set of reaction additions from a reference universal database that resolves dead-end metabolites and restores overall cellular viability (growth).

---

## 💻 5. Codebase Directory Map

```text
├── app.py                      # Master Streamlit dashboard & plotting logic
├── run_project.sh              # Bash script to boot up virtualenv, backend, & frontend
├── requirements.txt            # Python dependencies list
├── PROCESS_EXPLAINED.md        # Biological/user guide to the features
├── FLOW_GUIDE.md               # User & developer flow guide
├── PERFORMANCE_AND_TIMINGS.md  # Latencies, timeouts, and optimization parameters
│
├── core/                       # Scientific core components
│   ├── model_loader.py         # SBML parser, summary extractor, and file I/O
│   ├── diagnostics.py          # Math wrapper scripts running FBA and pFBA
│   ├── validation.py           # Elemental mass balancing & topological checks
│   └── genome_pipeline.py      # Homology annotation, EC mapping, & gap-filling
│
├── utils/                      # Root utility functions
│   └── solver_utils.py         # Solver detection & tolerance settings
│
└── backend/                    # FastAPI App Structure
    ├── main.py                 # Lifespan tasks, middleware, CORS, & dev server
    ├── exceptions.py           # Domain exceptions mapped to HTTP codes
    ├── api/
    │   └── routes.py           # REST endpoints, timeouts, and thread allocations
    ├── schemas/
    │   ├── requests.py         # Pydantic request body validators
    │   └── responses.py        # Pydantic serialization models
    ├── services/
    │   ├── model_registry.py   # Thread-safe in-memory model database with TTL eviction
    │   ├── model_service.py    # SBML upload handler & persistent default setting
    │   ├── fba_service.py      # Standard & parsimonious FBA execution contexts
    │   ├── fva_service.py      # Flux Variability Analysis runner
    │   ├── medium_service.py   # Medium configuration (presets / overlays)
    │   ├── production_service.py# Switch objective & Pareto envelope logic
    │   ├── optknock_service.py # Greedy strain design knockout heuristic
    │   └── growth_audit_service.py # 10-step zero-growth debugger
    └── utils/
        └── solve_utils.py      # Backend allowlist, timeouts, & validation helper
```
