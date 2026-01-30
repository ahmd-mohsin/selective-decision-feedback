# README.md - Physical Layer Module (Transmission)

## Overview

This module implements the complete physical layer transmission chain for a massive MIMO-OFDM system. It serves as the foundation for the Hybrid RC-Flow channel estimation framework by generating realistic wireless communication scenarios with ground truth data.

## What This Module Does

### 1. **Signal Generation & Modulation**
- Generates random binary data sequences
- Maps bits to QAM constellation symbols (16-QAM, 64-QAM supported)
- Creates OFDM grids with interleaved pilot and data symbols
- Normalizes constellation for unit average power

### 2. **Channel Modeling**
- Implements 3GPP CDL (Clustered Delay Line) channel models:
  - **CDL-A**: Mixed LOS/NLOS propagation
  - **CDL-B**: NLOS with rich scattering
  - **CDL-C**: NLOS moderate scattering
  - **CDL-D**: LOS dominant (Rician fading)
- Simulates multipath propagation with realistic angular spreads
- Applies Rayleigh/Rician fading based on channel type

### 3. **Wireless Impairments**
- Adds AWGN (Additive White Gaussian Noise) at specified SNR levels
- Applies frequency-selective fading across OFDM subcarriers
- Introduces channel estimation uncertainty

### 4. **Reception & Demodulation**
- **Zero-Forcing (ZF) Equalization**: Inverts channel matrix to recover symbols
- **MMSE Equalization**: Optimal linear equalization considering noise
- **Hard Decision Demodulation**: Slices soft symbols to nearest constellation points
- **Confidence Metric Calculation**: Computes Error Vector Magnitude (EVM) for each symbol

### 5. **Dataset Creation**
- Generates large-scale datasets with ground truth channel matrices
- Stores transmitted signals, received signals, pilot patterns, and channel responses
- Separates training, validation, and test sets

## System Architecture
```
Transmitter Flow:
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ Bit Stream  │───▶│ QAM Modulator│───▶│  OFDM Grid  │
└─────────────┘    └──────────────┘    └─────────────┘
                                              │
                                              ▼
                                        ┌─────────────┐
                                        │  Add Pilots │
                                        └─────────────┘

Channel:
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ OFDM Symbol │───▶│ CDL Channel  │───▶│   + AWGN    │
└─────────────┘    │   (H matrix) │    └─────────────┘
                   └──────────────┘

Receiver Flow:
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Received Y │───▶│ Equalization │───▶│ Hard Decision│
└─────────────┘    │   (ZF/MMSE)  │    └─────────────┘
                   └──────────────┘           │
                                              ▼
                                        ┌─────────────┐
                                        │ EVM & Conf. │
                                        └─────────────┘
```

## Key Components

### **SystemConfig** (`config.py`)
Central configuration class that defines:
- Array dimensions (Nr × Nt MIMO)
- Number of pilots (Np) and pilot density (α = Np/Nt)
- Modulation order (16-QAM, 64-QAM)
- SNR operating point
- Channel model selection

### **Modulator** (`modulator.py`)
Handles all transmitter-side operations:
- `generate_random_bits()`: Creates random binary sequences
- `bits_to_symbols()`: Gray-coded QAM mapping
- `generate_pilots()`: QPSK pilot generation
- `create_ofdm_grid()`: Pilot-data multiplexing

### **CDLChannel** (`channel.py`)
Implements 3GPP-compliant channel models:
- `generate_response()`: Creates Nr × Nt channel matrix H
- `apply()`: Convolves signal with channel and adds noise
- Multi-path modeling with angle-of-arrival/departure
- Rician K-factor for LOS channels

### **Receiver** (`receiver.py`)
Performs signal recovery:
- `zero_forcing_equalization()`: Simple inversion (works when Np ≥ Nt)
- `mmse_equalization()`: Wiener filter (better for low SNR)
- `process_received_signal()`: End-to-end demodulation pipeline

### **Demodulator** (`demodulator.py`)
Symbol-level processing:
- `hard_decision()`: Minimum Euclidean distance detection
- `calculate_evm()`: Error vector magnitude for confidence
- `calculate_confidence()`: Thresholding for pseudo-pilot selection

### **DatasetGenerator** (`dataset_generator.py`)
Orchestrates the entire pipeline:
- Generates thousands of independent channel realizations
- Saves ground truth H, transmitted/received signals, pilot masks
- Creates train/val/test splits

## Why This Matters for Decision-Directed Extension

This module produces the critical **confidence metrics** that enable the Decision-Directed feedback loop:

1. **EVM Calculation**: Measures `||y_soft - y_hard||` for each demodulated symbol
2. **Confidence Thresholding**: Selects only reliable symbols (low EVM)
3. **Pseudo-Pilot Creation**: High-confidence symbols become additional "measurements"

The confidence mask will be fed to Phase 2 of the Hybrid RC-Flow to augment the pilot matrix.

## File Structure
```
transmission/
├── __init__.py                 # Module exports
├── config.py                   # System parameters
├── constellation.py            # QAM constellation generation
├── modulator.py               # Transmitter
├── channel.py                 # Wireless channel models
├── demodulator.py             # Symbol detection
├── receiver.py                # Equalization + demodulation
└── dataset_generator.py       # End-to-end dataset creation
```

## Installation
```bash
bash setup.sh
```

This will:
- Create a conda environment with Python 3.10
- Install NumPy, SciPy, Matplotlib
- Create necessary directories

## Running Dataset Generation

### Quick Start
```bash
bash run_dataset_generation.sh
```

### Manual Execution
```bash
conda activate hybrid-rcflow
python main_generate_dataset.py
```

### Custom Configuration
Edit `main_generate_dataset.py` to adjust:
```python
config = SystemConfig(
    Nr=16,              # Receive antennas
    Nt=64,              # Transmit antennas
    Np=38,              # Number of pilots (α = 0.59)
    snr_db=10.0,        # Operating SNR
    channel_model='CDL-C',  # Channel type
    num_samples=10000   # Dataset size
)
```

## Output Format

Each dataset sample is a dictionary containing:
```python
{
    'H_true': np.ndarray,          # Ground truth channel (Nr × Nt)
    'transmitted_grid': np.ndarray, # Tx OFDM symbols (Nt × T)
    'received_grid': np.ndarray,    # Rx OFDM symbols (Nr × T)
    'pilot_mask': np.ndarray,       # Boolean mask (Nt × T)
    'pilots': np.ndarray,           # Pilot symbols (Np,)
    'data_bits': np.ndarray,        # Original bits
    'data_symbols': np.ndarray,     # Modulated data
    'snr_db': float                # Operating SNR
}
```

## Understanding the Confidence Mechanism

### How EVM Captures Reliability

When a symbol is decoded:
1. **Soft Symbol**: `y_soft = H_inv @ y_received` (equalized)
2. **Hard Decision**: `y_hard = argmin ||y_soft - c_i||` (nearest constellation point)
3. **EVM**: `evm = ||y_soft - y_hard||`

**Low EVM** → Symbol landed close to constellation point → High confidence
**High EVM** → Symbol in decision region boundary → Low confidence (could be wrong)

### Confidence Thresholding
```python
threshold = np.percentile(evm, 25)  # Take top 75% reliable symbols
confidence_mask = evm < threshold
```

Only symbols with `confidence_mask = True` will be used as pseudo-pilots in Phase 2.

## Key Parameters Explained

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| Nr        | 16            | Base station receive antennas |
| Nt        | 64            | User equipment transmit antennas |
| Np        | 38            | Number of orthogonal pilots |
| α         | 0.59          | Pilot density (Np/Nt) |
| SNR       | -10 to 30 dB  | Signal-to-noise ratio range |
| Mod Order | 16, 64        | QAM constellation size |

## Next Steps

After generating the dataset, the next modules will be:

1. **RC-Flow Module**: Generative channel estimation using flow matching
2. **Decision-Directed Module**: Pseudo-pilot selection and augmentation
3. **Training Module**: End-to-end training of the hybrid system

## Troubleshooting

**Issue**: `LinAlgError` during equalization
**Solution**: Increase pilot density (Np) or switch to MMSE equalization

**Issue**: Low confidence mask (< 50% symbols selected)
**Solution**: Increase SNR or reduce modulation order

**Issue**: Out of memory during dataset generation
**Solution**: Reduce `num_samples` or process in batches

## Mathematical Notation

- **H**: Channel matrix (Nr × Nt)
- **Y**: Received signal matrix (Nr × T)
- **P**: Pilot matrix (Nt × Np)
- **α**: Pilot density Np/Nt
- **EVM**: Error Vector Magnitude ||y_soft - y_hard||
- **SNR**: 10 log₁₀(Signal Power / Noise Power)

## References

This implementation follows:
- 3GPP TR 38.901 for CDL channel models
- RC-Flow paper (arXiv:2601.15767v2) for system setup
- Classical decision-directed equalization theory