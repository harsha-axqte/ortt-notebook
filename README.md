# Object-Relative Temporal Theory (ORTT) Simulations

This repository contains **fully reproducible simulations** for the **Object-Relative Temporal Theory (ORTT)**, developed to explore **emergent, object-relative time** in hybrid quantum–classical systems.

**Paper reference:**  
Harsha Vardhan Routhu, *"Object-Relative Temporal Theory (ORTT): Redefining the Ontology of Time"*, submitted to [Journal/PRX].

---

## 1. Overview

### What is ORTT?

ORTT proposes that **time is not universal**, but emerges **relationally** from entropy dynamics and subsystem interactions.  

**Key principles:**

- **Relational Time:** Each subsystem has its own temporal parameter, $\tau$, derived from internal and environmental entropy changes.  
- **Entropy–Time Coupling:** Effective time increments depend on the rate of entropy production.  
- **Interaction Dependence:** Coupled systems dynamically adjust relative time scales via interactions.

ORTT provides a **quantitative framework** to predict measurable deviations in quantum and classical dynamics due to entropy-driven time scaling.

---

## 2. Core Equations Implemented

### Quantum Subsystems

**Object-relative quantum time increment:**

$$
\Delta \tau_Q = \Delta t \left[ 1 + \alpha_Q f\left(\frac{dS}{dt}\right) \right]
$$

**Quantum evolution (von Neumann + Lindblad):**

$$
\frac{d\rho}{d\tau_Q} = -\frac{i}{\hbar}[H, \rho] + \mathcal{L}(\rho)
$$

Where:

- $\rho$ = density matrix  
- $S[\rho] = -\mathrm{Tr}(\rho \log \rho)$ = von Neumann entropy  
- $H$ = Hamiltonian + interaction-dependent feedback  
- $\mathcal{L}(\rho)$ = Lindblad dissipator  
- $f$ = bounded monotonic function ($\tanh$ in code)  
- $\alpha_Q$ = quantum coupling constant  

---

### Classical Subsystems

**Classical object-relative time increment:**

$$
\Delta \tau_C = \Delta t \left[ 1 + \alpha_C g(x) \right]
$$

**Classical evolution:**

$$
\frac{dx}{d\tau_C} = F(x, \rho)
$$

Where:

- $x(t) = [V_m, g]$ = classical state variables (membrane potential, gating)  
- $g(x)$ = function of subsystem observables  
- $F(x, \rho)$ = dynamics including quantum feedback  

---

### Interaction Scaling

For coupled subsystems $O_1, O_2$:

$$
\Delta \tau_{O_i} \mapsto \Delta \tau_{O_i} \cdot \left[1 + \beta_{ij} h(O_i, O_j)\right]
$$

Where:

- $\beta_{ij}$ = interaction strength  
- $h(O_i, O_j)$ = functional of correlations or shared entropy  

---

## 3. Repository Structure
```bash

ORTT_Simulations/
├── ORTT_simulation.ipynb # Main Jupyter notebook
├── figures/ 
├── requirements.txt # Python dependencies
├── python/
├── README.md 

```

---

## 4. Code Details

### Quantum Evolution

- Hamiltonian: `H0` + feedback term proportional to $V_m$ & $g$  
- Lindblad operator `L` introduces dissipation  
- Uses `expm` for unitary evolution and explicit Lindblad step  
- Function: `evolve_quantum(rho, H0, Vm, g, dtQ)`

### Classical Neuron Dynamics

- Membrane potential $V_m$ and gating variable $g$  
- Leak current + quantum feedback current  
- Integrated with `solve_ivp` using RK45  
- Function: `classical_rhs(t, y, rho)`

### Entropy & Time Scaling

- Quantum von Neumann entropy: `von_neumann_entropy(rho)`  
- ORTT $\tau$ scaling vs control  
- Functions `trace_distance` and `fidelity` compare ORTT and control quantum states  

### Simulation Runner

- `run_sim(use_ortt=True/False)` runs hybrid dynamics  
- Returns classical & quantum state evolution, $\tau$ increments, Bloch components  

### Post-Processing

- $\Delta V_m$ (ORTT – Control)  
- Trace distance and fidelity²  
- Cumulative $\tau$ plots  
- Bloch sphere & phase-space plots  

---

## 5. Reproducing Figures

Run the notebook top-to-bottom:

```bash
git clone https://github.com/harsha-axqte/ortt-notebook.git
cd ORTT_Simulations
pip install -r requirements.txt
jupyter notebook ORTT_simulation.ipynb

```
Figures produced:

| Filename |Description |
|------|----------------|
| fig_delta_vm.pdf |	Membrane potential difference |
| fig_quantum_diff.pdf |	Quantum state metrics (trace distance, fidelity²)|
|fig_cumulative_tau.pdf |	Cumulative object-relative times|
|fig_bloch_traj.pdf |	Bloch sphere trajectory|
|fig_vm_pop.pdf |	Neuron potential & qubit populations|
|fig_phase_traj.pdf |	Phase-space trajectory|
|fig_phase_heatmap.pdf|	Phase-space density heatmap|

All figures are generated automatically.

## 6. Zenodo Archival
This repository has been archived on Zenodo for reproducibility:

DOI: https://doi.org/10.5281/zenodo.17229366

## 7. Requirements
Python ≥ 3.9

Packages: NumPy, SciPy, Matplotlib, Jupyter Notebook

Install dependencies:

```bash
pip install -r requirements.txt
```

## 8. License
This code is licensed under CC-BY 4.0 — reuse, adapt, and distribute with proper attribution.

## 9. Contact
Harsha Vardhan Routhu
AXQTE, Hyderabad, India
Email: harsha@AXQTE.com

### Notes for Reviewers / Readers
Background $\Delta t$: Used only for numerical integration; physical predictions depend on $\tau$ ratios ($\tau_Q/\tau_C$).

Physical Meaning: Divergences between ORTT and control illustrate measurable consequences of entropy-driven time.

### Extensibility: Notebook can be adapted to more complex neuron-qubit networks, different Hamiltonians, and classical subsystems.
