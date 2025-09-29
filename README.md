Object-Relative Temporal Theory (ORTT) Simulations

This repository contains fully reproducible simulations for the Object-Relative Temporal Theory (ORTT), developed to explore emergent, object-relative time in hybrid quantum–classical systems.

Paper reference:
Harsha Vardhan Routhu, "Object-Relative Temporal Theory (ORTT): Redefining the Ontology of Time", submitted to [Journal/PRX].

1. Overview
What is ORTT?

ORTT proposes that time is not universal, but emerges relationally from entropy dynamics and subsystem interactions.

Key principles:

Relational Time: Each subsystem has its own temporal parameter, $\tau$, derived from internal and environmental entropy changes.

Entropy–Time Coupling: Effective time increments depend on the rate of entropy production.

Interaction Dependence: Coupled systems dynamically adjust relative time scales via interactions.

ORTT provides a quantitative framework to predict measurable deviations in quantum and classical dynamics due to entropy-driven time scaling.

2. Core Equations Implemented
Quantum Subsystems

Object-relative quantum time increment:

Δ
𝜏
𝑄
=
Δ
𝑡
[
1
+
𝛼
𝑄
𝑓
 ⁣
(
𝑑
𝑆
𝑑
𝑡
)
]
Δτ
Q
	​

=Δt[1+α
Q
	​

f(
dt
dS
	​

)]

Quantum evolution (von Neumann + Lindblad):

𝑑
𝜌
𝑑
𝜏
𝑄
=
−
𝑖
ℏ
[
𝐻
,
𝜌
]
+
𝐿
(
𝜌
)
dτ
Q
	​

dρ
	​

=−
ℏ
i
	​

[H,ρ]+L(ρ)

Where:

$\rho$ = density matrix

$S[\rho] = -\mathrm{Tr}(\rho \log \rho)$ = von Neumann entropy

$H$ = Hamiltonian + interaction-dependent feedback

$\mathcal{L}(\rho)$ = Lindblad dissipator

$f$ = bounded monotonic function ($\tanh$ in code)

$\alpha_Q$ = quantum coupling constant

Classical Subsystems

Classical object-relative time increment:

Δ
𝜏
𝐶
=
Δ
𝑡
[
1
+
𝛼
𝐶
𝑔
(
𝑥
)
]
Δτ
C
	​

=Δt[1+α
C
	​

g(x)]

Classical evolution:

𝑑
𝑥
𝑑
𝜏
𝐶
=
𝐹
(
𝑥
,
𝜌
)
dτ
C
	​

dx
	​

=F(x,ρ)

Where:

$x(t) = [V_m, g]$ = classical state variables (membrane potential, gating)

$g(x)$ = function of subsystem observables

$F(x, \rho)$ = dynamics including quantum feedback

Interaction Scaling

For coupled subsystems $O_1, O_2$:

Δ
𝜏
𝑂
𝑖
↦
Δ
𝜏
𝑂
𝑖
⋅
[
1
+
𝛽
𝑖
𝑗
ℎ
(
𝑂
𝑖
,
𝑂
𝑗
)
]
Δτ
O
i
	​

	​

↦Δτ
O
i
	​

	​

⋅[1+β
ij
	​

h(O
i
	​

,O
j
	​

)]

Where:

$\beta_{ij}$ = interaction strength

$h(O_i, O_j)$ = functional of correlations or shared entropy

3. Repository Structure
ORTT_Simulations/
├── ORTT_simulation.ipynb  # Main Jupyter notebook
├── figures/               # Output figures (PDFs)
├── requirements.txt       # Python dependencies
├── README.md              # This file

4. Code Details
Quantum Evolution

Hamiltonian: H0 + feedback term proportional to $V_m$ & $g$

Lindblad operator L introduces dissipation

Uses expm for unitary evolution and explicit Lindblad step

Function: evolve_quantum(rho, H0, Vm, g, dtQ)

Classical Neuron Dynamics

Membrane potential $V_m$ and gating variable $g$

Leak current + quantum feedback current

Integrated with solve_ivp using RK45

Function: classical_rhs(t, y, rho)

Entropy & Time Scaling

Quantum von Neumann entropy: von_neumann_entropy(rho)

ORTT $\tau$ scaling vs control

Functions trace_distance and fidelity compare ORTT and control quantum states

Simulation Runner

run_sim(use_ortt=True/False) runs hybrid dynamics

Returns classical & quantum state evolution, $\tau$ increments, Bloch components

Post-Processing

$\Delta V_m$ (ORTT – Control)

Trace distance and fidelity²

Cumulative $\tau$ plots

Bloch sphere & phase-space plots

5. Reproducing Figures

Run the notebook top-to-bottom:

git clone https://github.com/harsha-axqte/ortt-notebook.git
cd ORTT_Simulations
pip install -r requirements.txt
jupyter notebook ORTT_simulation.ipynb


Figures produced:

Filename	Description
fig_delta_vm.pdf	Membrane potential difference
fig_quantum_diff.pdf	Quantum state metrics (trace distance, fidelity²)
fig_cumulative_tau.pdf	Cumulative object-relative times
fig_bloch_traj.pdf	Bloch sphere trajectory
fig_vm_pop.pdf	Neuron potential & qubit populations
fig_phase_traj.pdf	Phase-space trajectory
fig_phase_heatmap.pdf	Phase-space density heatmap

All figures are generated automatically.

6. Zenodo Archival

This repository has been archived on Zenodo for reproducibility:

DOI: https://doi.org/XXXX/zenodo.XXXXX

7. Requirements

Python ≥ 3.9

Packages: NumPy, SciPy, Matplotlib, Jupyter Notebook

Install dependencies:

pip install -r requirements.txt

8. License

CC-BY 4.0 — reuse, adapt, and distribute with proper attribution.

9. Contact

Harsha Vardhan Routhu
AXQTE, Hyderabad, India
Email: harsha@AXQTE.com

Notes for Reviewers / Readers

Background $\Delta t$: Used only for numerical integration; physical predictions depend on $\tau$ ratios ($\tau_Q/\tau_C$).

Physical Meaning: Divergences between ORTT and control illustrate measurable consequences of entropy-driven time.

Extensibility: Adaptable to more complex neuron-qubit networks, different Hamiltonians, and classical subsystems.
