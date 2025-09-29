# Object-Relative Temporal Theory (ORTT) Simulations

This repository contains **fully reproducible simulations** for the Object-Relative Temporal Theory (ORTT), developed to explore **emergent, object-relative time** in hybrid quantum–classical systems.

**Paper reference:**  
Harsha Vardhan Routhu, *"Object-Relative Temporal Theory (ORTT): Redefining the Ontology of Time"*, submitted to [Journal/PRX].

---

## 1. Overview

### What is ORTT?

ORTT proposes that **time is not universal**, but emerges **relationally** from entropy dynamics and subsystem interactions. Key ideas:

1. **Relational Time:** Each physical subsystem has its own temporal parameter (τ) based on internal and environmental entropy changes.
2. **Entropy–Time Coupling:** The effective time increment depends on the rate of entropy production.
3. **Interaction Dependence:** Coupled systems dynamically adjust their relative time scales based on interactions.

ORTT provides a **quantitative framework** to predict measurable deviations in quantum and classical dynamics due to entropy-driven time scaling.

---

### Core Equations Implemented

#### Quantum Subsystems:

The object-relative quantum time increment:
\[
\Delta \tau_Q = \Delta t \left[ 1 + \alpha_Q f\!\left( \frac{dS}{dt} \right) \right]
\]

Evolution (von Neumann + Lindblad):
\[
\frac{d\rho}{d\tau_Q} = -\frac{i}{\hbar}[H, \rho] + \mathcal{L}(\rho)
\]

Where:
- \(\rho\) = density matrix
- \(S[\rho] = -\mathrm{Tr}(\rho \log \rho)\) = von Neumann entropy
- \(H\) = Hamiltonian + interaction-dependent feedback
- \(\mathcal{L}(\rho)\) = Lindblad dissipator
- \(f\) = bounded monotonic function (implemented as \(\tanh\))
- \(\alpha_Q\) = quantum coupling constant

#### Classical Subsystems:

The classical object-relative time increment:
\[
\Delta \tau_C = \Delta t \left[ 1 + \alpha_C g(x) \right]
\]

Classical evolution:
\[
\frac{dx}{d\tau_C} = F(x, \rho)
\]

Where:
- \(x(t) = [V_m, g]\) = state variables (membrane potential, gating)
- \(g(x)\) = function of subsystem observables (implemented as voltage + gating feedback)
- \(F(x, \rho)\) = dynamics including quantum feedback

#### Interaction Scaling:

For coupled subsystems \(O_1, O_2\):
\[
\Delta \tau_{O_i} \mapsto \Delta \tau_{O_i} \cdot \left[1 + \beta_{ij} h(O_i, O_j)\right]
\]

Where:
- \(\beta_{ij}\) = interaction strength
- \(h(O_i, O_j)\) = functional of correlations or shared entropy

---

## 2. Repository Structure

ORTT_Simulations/
├── ORTT_simulation.ipynb # Main Jupyter notebook
├── figures/ # Output figures (PDFs)
├── requirements.txt # Python dependencies
├── README.md # This file

markdown
Copy code

---

## 3. Code Details

### Simulation Components:

1. **Quantum Evolution**
   - Hamiltonian: `H0` + feedback term proportional to Vm & gating variable
   - Lindblad operator: `L` introduces dissipation
   - Uses `expm` for unitary evolution and explicit Lindblad step
   - Function: `evolve_quantum(rho, H0, Vm, g, dtQ)`

2. **Classical Neuron Dynamics**
   - Membrane potential \(V_m\) and gating variable \(g\)
   - Leak current + quantum feedback current
   - Integrated with `solve_ivp` using RK45
   - Function: `classical_rhs(t, y, rho)`

3. **Entropy & Time Scaling**
   - Quantum von Neumann entropy: `von_neumann_entropy(rho)`
   - τ scaling: ORTT vs Control (τ = Δt × scaling factor)
   - Functions `trace_distance` and `fidelity` compare ORTT and control quantum states

4. **Simulation Runner**
   - `run_sim(use_ortt=True/False)` runs hybrid dynamics
   - Returns classical & quantum state evolution, τ increments, Bloch components

5. **Post-Processing**
   - ΔVm (ORTT – Control)
   - Trace distance and fidelity²
   - Cumulative τ plots
   - Bloch sphere and phase-space plots

---

## 4. Reproducing Figures

Run the notebook top-to-bottom:

```bash
git clone https://github.com/harsha-axqte/ortt-notebook.git
cd ORTT_Simulations
pip install -r requirements.txt
jupyter notebook ORTT_simulation.ipynb
Figures produced:

fig_delta_vm.pdf — Membrane potential difference

fig_quantum_diff.pdf — Quantum state metrics

fig_cumulative_tau.pdf — Cumulative object-relative times

fig_bloch_traj.pdf — Bloch sphere trajectory

fig_vm_pop.pdf — Neuron potential & qubit populations

fig_phase_traj.pdf — Phase-space trajectory

fig_phase_heatmap.pdf — Phase-space density heatmap

All figures are generated automatically when running the notebook.

5. Zenodo Archival
This repository has been archived on Zenodo for reproducibility and citation:
DOI: https://doi.org/XXXX/zenodo.XXXXX

6. Requirements
Python ≥ 3.9

NumPy, SciPy, Matplotlib

Optional: Jupyter Notebook / JupyterLab

Install all dependencies:
pip install -r requirements.txt
7. License
This code is licensed under CC-BY 4.0. You may reuse, adapt, and distribute with proper attribution.

8. Contact
Harsha Vardhan Routhu
AXQTE, Hyderabad, India
Email: harsha@AXQTE.com

Notes for Reviewers / Readers

Background Δt: Used only for numerical integration; physical predictions depend on τ ratios (τQ/τC).

Physical Meaning: Divergences between ORTT and control illustrate measurable consequences of entropy-driven time.

Extensibility: The notebook can be adapted to more complex neuron-qubit networks, different Hamiltonians, and classical subsystems.
