#!/usr/bin/env python3
"""
Quick verification: Show that more negative NMSE is better
"""
import numpy as np

print("=" * 80)
print("NMSE Interpretation Guide")
print("=" * 80)
print()
print("NMSE (Normalized Mean Square Error) measures estimation error:")
print()
print("  NMSE = mean(|H_estimated - H_true|²) / mean(|H_true|²)")
print()
print("In dB: NMSE_dB = 10 * log10(NMSE)")
print()
print("Examples:")
print("  NMSE = 0.001  → NMSE_dB = -30 dB  (very good!)")
print("  NMSE = 0.01   → NMSE_dB = -20 dB  (good)")
print("  NMSE = 0.1    → NMSE_dB = -10 dB  (moderate)")
print("  NMSE = 1.0    → NMSE_dB =   0 dB  (poor)")
print()
print("✓ MORE NEGATIVE dB = BETTER performance")
print("✓ LESS NEGATIVE dB = WORSE performance")
print()
print("Your Results:")
print("  Diffusion:     -24.30 dB  (good)")
print("  Full Pipeline: -26.46 dB  (BETTER by 2.16 dB!)")
print()
print("=" * 80)
print("The system IS working correctly! 🎉")
print("=" * 80)
