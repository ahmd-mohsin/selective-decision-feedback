Here is the comprehensive `README.md` and the detailed File Structure guide. You can copy-paste this entire response into a new chat with ChatGPT (or any coding assistant) to get it to write the exact code you need.

---

# 1. Project Masterplan & README

**Project Title:** **Hybrid RC-Flow: Decision-Directed Generative MIMO Channel Estimation**

## 📖 Abstract

This project bridges the gap between state-of-the-art Generative AI (RC-Flow) and classical Adaptive Signal Processing (Decision-Directed Estimation).

Current generative methods for channel estimation (like the "Recursive Flow" paper) treat the problem as a "Cold Start" ill-posed inverse problem, relying solely on sparse pilot signals. This ignores the massive information content available in the data payload. This project extends the RC-Flow framework by implementing a **Decision-Directed (DD)** loop. By demodulating the data and filtering for high-confidence symbols, we convert the payload into "Pseudo-Pilots," effectively turning the problem from *under-determined* to *over-determined* and achieving superior estimation accuracy.

## 🚀 System Architecture

The system operates in three distinct phases:

1. **Phase 1: Cold Start (Generative Prior)**
* **Input:** Noisy received signal () + Known Pilots ().
* **Engine:** RC-Flow (Flow Matching Network + Physics-Aware Projector).
* **Goal:** Generate a "good enough" initial estimate () to open the "Eye Diagram."


2. **Phase 2: Decision Directed (Tracking)**
* **Input:**  + .
* **Action:** Zero-Forcing Equalization and Hard Slicing (Demodulation).
* **Confidence Logic:** Calculate the Euclidean distance between Soft Symbols and Hard Decisions. Only symbols with error  are selected as "Pseudo-Pilots."
* **Goal:** Extract "ground truth" data from the payload.


3. **Phase 3: Recursive Refinement (Hybrid)**
* **Input:**  + Augmented Reference Signals (Pilots + Pseudo-Pilots).
* **Action:** Restart the RC-Flow solver. The "Physics-Aware Projector" now constrains the solution to match *both* the original pilots and the decoded data.
* **Result:** A highly accurate channel estimate effectively using the entire packet energy.



---

# 2. Detailed File Structure & Implementation Guide

Use this structure to organize the codebase. When asking an AI to write code, refer to these file descriptions.

## Root Directory

* `main.py`: The orchestrator. It initializes the channel, runs the simulation loop, triggers RC-Flow, then DD, then Refined RC-Flow.
* `config.yaml`: Central configuration (SNR, FFT size, Pilot patterns, Confidence thresholds).
* `environment.yml`: Conda environment dependencies.

## Module 1: `transmission/` (The Physical Layer)

*This module simulates the "Real World" physics and the transceiver hardware.*

* **`modulator.py`**
* **Role:** The Transmitter (Tx).
* **Key Functions:**
* `bits_to_symbols()`: Maps binary strings to QAM constellations (16-QAM, 64-QAM).
* `generate_ofdm_grid()`: Arranges Pilots and Data into the Time-Frequency grid.




* **`channel.py`**
* **Role:** The Wireless Environment.
* **Key Functions:**
* `generate_response()`: Creates the ground truth  (Rayleigh/Rician/CDL models).
* `apply()`: Performs . Adds fading and AWGN noise.




* **`receiver.py`**
* **Role:** The Receiver (Rx) Front-end.
* **Key Functions:**
* `equalize()`: Performs Zero-Forcing (ZF) or MMSE equalization using an estimated .
* `demodulate()`: Slices soft symbols to the nearest hard constellation point.
* **Crucial:** Calculates the **Confidence Metric** (Error Vector Magnitude) for every symbol.





## Module 2: `rc_flow/` (The AI Engine)

*This module implements the "Recursive Flow" paper logic.*

* **`network.py`**
* **Role:** The Flow Matching Neural Network.
* **Details:** A U-Net or ResNet that takes  as input and outputs the vector field . This is the "Prior" that knows what valid channels look like.


* **`projector.py`**
* **Role:** The Physics Consistency Enforcer.
* **Algorithm:** Solves .
* **Why it's unique:** It must accept a dynamic Mask.
* *Pass 1:* Mask covers only Pilots.
* *Pass 2:* Mask covers Pilots + Pseudo-Pilots.




* **`ode_solver.py`**
* **Role:** The Iterative Solver.
* **Logic:** Implements the Euler steps + Recursive Anchor Refinement described in the paper. It calls `network.py` to move and `projector.py` to stay on track.



## Module 3: `decision_directed/` (The Extension)

*This module implements the Professor's "Chapter 4" requirements.*

* **`selector.py`**
* **Role:** The Gatekeeper.
* **Logic:** Takes the decoded symbols and their confidence scores.
* **Function:** `select_reliable_symbols(hard_syms, errors, threshold)`. Returns a boolean mask and the values for the "Pseudo-Pilots."


* **`adapter.py`** (Optional wrapper)
* **Role:** Formats the "Pseudo-Pilots" so they can be fed back into the `rc_flow/projector.py`. Merges the original pilot mask with the new data mask.



