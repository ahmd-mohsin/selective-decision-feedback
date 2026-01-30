# RC-Flow + Decision Directed (DD) Channel Estimation

This project implements a hybrid MIMO Channel Estimation framework that combines **Generative AI (RC-Flow)** with classical **Decision-Directed (DD)** adaptation.

## 🚀 The Core Idea
1.  **Cold Start (RC-Flow):** Uses a Flow Matching generative model to solve the ill-posed inverse problem using only sparse pilots.
2.  **Tracking (Decision Directed):** Uses the initial estimate to decode payload data.
3.  **Refinement (Hybrid Loop):** "Confident" decoded symbols are treated as new pilots to convert the problem from *under-determined* to *over-determined*, drastically improving accuracy.

## 📂 Directory Structure

```text
project_root/
├── config.yaml                 # Centralized Simulation Parameters
├── main.py                     # Entry point (Runs the full loop)
├── transmission/               # [PART 1] The Physical Layer Simulation
│   ├── modulator.py            # Bits -> QAM -> OFDM Grid (Pilots allocation)
│   ├── channel.py              # Wireless Fading (Rayleigh/CDL) + Noise
│   └── receiver.py             # Equalizer, Slicer, and Confidence Metrics
├── rc_flow/                    # [PART 2] The Generative AI Solver
│   ├── network.py              # Flow Matching U-Net
│   ├── ode_solver.py           # Recursive Euler Solver
│   └── projector.py            # Physics-Aware Proximal Projection
└── decision_directed/          # [PART 3] The Adapter
    └── strategy.py             # Logic to merge AI priors with DD measurements