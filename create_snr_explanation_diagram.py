import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))

# ==================== Subplot 1: SNR Definition ====================
ax1 = fig.add_subplot(2, 3, 1)
ax1.axis('off')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)

# Title
ax1.text(5, 9.5, 'SNR Definition', fontsize=16, fontweight='bold', ha='center')

# Signal power
signal_rect = FancyBboxPatch((1, 6.5), 3, 1.5, boxstyle="round,pad=0.1", 
                             edgecolor='green', facecolor='lightgreen', linewidth=2)
ax1.add_patch(signal_rect)
ax1.text(2.5, 7.25, 'Signal Power = 1.0', fontsize=11, ha='center', fontweight='bold')

# Arrow
ax1.annotate('', xy=(5, 7.25), xytext=(4, 7.25),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# SNR formula
formula_rect = FancyBboxPatch((5, 6.5), 4, 1.5, boxstyle="round,pad=0.1",
                             edgecolor='blue', facecolor='lightblue', linewidth=2)
ax1.add_patch(formula_rect)
ax1.text(7, 7.5, 'SNR = 10 log₁₀(1/σ²)', fontsize=10, ha='center', fontweight='bold')
ax1.text(7, 7.0, 'σ² = 10^(-SNR/10)', fontsize=9, ha='center')

# Noise power
noise_rect = FancyBboxPatch((1, 4), 3, 1.5, boxstyle="round,pad=0.1",
                           edgecolor='red', facecolor='lightcoral', linewidth=2)
ax1.add_patch(noise_rect)
ax1.text(2.5, 4.75, 'Noise Variance = σ²', fontsize=11, ha='center', fontweight='bold')

# Examples
ax1.text(5, 3, 'Examples:', fontsize=11, fontweight='bold')
ax1.text(5, 2.3, 'SNR = 10 dB → σ² = 0.100', fontsize=10)
ax1.text(5, 1.7, 'SNR = 20 dB → σ² = 0.010', fontsize=10)
ax1.text(5, 1.1, 'SNR = 30 dB → σ² = 0.001', fontsize=10)

# ==================== Subplot 2: Noise Variance vs SNR ====================
ax2 = fig.add_subplot(2, 3, 2)
snr_range = np.arange(0, 35, 1)
noise_var = 10**(-snr_range/10)

ax2.semilogy(snr_range, noise_var, 'b-', linewidth=3, label='Noise Variance')
ax2.scatter([5, 10, 15, 20, 25, 30], 10**(-np.array([5, 10, 15, 20, 25, 30])/10), 
           s=150, c='red', zorder=5, edgecolors='black', linewidth=2)

# Add annotations for key points
for snr_val in [5, 10, 20, 30]:
    nv = 10**(-snr_val/10)
    ax2.annotate(f'{nv:.4f}', xy=(snr_val, nv), xytext=(snr_val+2, nv*1.5),
                fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

ax2.grid(True, alpha=0.3)
ax2.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Noise Variance (σ²)', fontsize=12, fontweight='bold')
ax2.set_title('Noise Power Added vs SNR', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)

# ==================== Subplot 3: NMSE Definition ====================
ax3 = fig.add_subplot(2, 3, 3)
ax3.axis('off')
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)

# Title
ax3.text(5, 9.5, 'NMSE Definition', fontsize=16, fontweight='bold', ha='center')

# True channel
true_rect = FancyBboxPatch((1, 7), 3, 1.2, boxstyle="round,pad=0.1",
                          edgecolor='purple', facecolor='plum', linewidth=2)
ax3.add_patch(true_rect)
ax3.text(2.5, 7.6, 'H_true (Ground Truth)', fontsize=10, ha='center', fontweight='bold')

# Estimated channel
est_rect = FancyBboxPatch((6, 7), 3, 1.2, boxstyle="round,pad=0.1",
                         edgecolor='orange', facecolor='moccasin', linewidth=2)
ax3.add_patch(est_rect)
ax3.text(7.5, 7.6, 'H_estimated', fontsize=10, ha='center', fontweight='bold')

# Arrow and error
ax3.annotate('', xy=(6, 7.6), xytext=(4, 7.6),
            arrowprops=dict(arrowstyle='<->', lw=2, color='red'))
ax3.text(5, 8.2, 'Error', fontsize=10, ha='center', color='red', fontweight='bold')

# NMSE formula
formula_box = FancyBboxPatch((2, 5), 6, 1.5, boxstyle="round,pad=0.1",
                            edgecolor='darkblue', facecolor='lightblue', linewidth=2)
ax3.add_patch(formula_box)
ax3.text(5, 6.0, 'NMSE = ||H_est - H_true||² / ||H_true||²', fontsize=9, ha='center', fontweight='bold')
ax3.text(5, 5.5, 'NMSE_dB = 10 log₁₀(NMSE)', fontsize=9, ha='center')

# Example
ax3.text(5, 3.8, 'Example:', fontsize=11, fontweight='bold')
ax3.text(5, 3.2, 'NMSE = -10 dB → Error is 10% of signal', fontsize=10)
ax3.text(5, 2.6, 'NMSE = -20 dB → Error is 1% of signal', fontsize=10)
ax3.text(5, 2.0, 'NMSE = -30 dB → Error is 0.1% of signal', fontsize=10)

ax3.text(5, 0.8, '⚠ NMSE is vs H_true, NOT vs baseline!', 
        fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# ==================== Subplot 4: Transmission System ====================
ax4 = fig.add_subplot(2, 3, 4)
ax4.axis('off')
ax4.set_xlim(0, 12)
ax4.set_ylim(0, 8)

# Title
ax4.text(6, 7.5, 'Transmission System', fontsize=14, fontweight='bold', ha='center')

# Transmitter
tx_rect = FancyBboxPatch((0.5, 5), 2, 1.5, boxstyle="round,pad=0.1",
                        edgecolor='green', facecolor='lightgreen', linewidth=2)
ax4.add_patch(tx_rect)
ax4.text(1.5, 5.75, 'Transmit\nX (symbols)', fontsize=9, ha='center', fontweight='bold')

# Channel
channel_rect = FancyBboxPatch((3.5, 5), 2, 1.5, boxstyle="round,pad=0.1",
                             edgecolor='blue', facecolor='lightblue', linewidth=2)
ax4.add_patch(channel_rect)
ax4.text(4.5, 5.75, 'Channel\nH', fontsize=9, ha='center', fontweight='bold')

# Noise
ax4.annotate('+ Noise\n(σ² from SNR)', xy=(4.5, 4.5), fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

# Receiver
rx_rect = FancyBboxPatch((6.5, 5), 2, 1.5, boxstyle="round,pad=0.1",
                        edgecolor='orange', facecolor='moccasin', linewidth=2)
ax4.add_patch(rx_rect)
ax4.text(7.5, 5.75, 'Receive\nY', fontsize=9, ha='center', fontweight='bold')

# Estimator
est_rect = FancyBboxPatch((9.5, 5), 2, 1.5, boxstyle="round,pad=0.1",
                         edgecolor='purple', facecolor='plum', linewidth=2)
ax4.add_patch(est_rect)
ax4.text(10.5, 5.75, 'Estimate\nĤ', fontsize=9, ha='center', fontweight='bold')

# Arrows
ax4.annotate('', xy=(3.5, 5.75), xytext=(2.5, 5.75),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax4.annotate('', xy=(6.5, 5.75), xytext=(5.5, 5.75),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax4.annotate('', xy=(9.5, 5.75), xytext=(8.5, 5.75),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Formula
ax4.text(6, 3.5, 'Received Signal:', fontsize=11, fontweight='bold', ha='center')
ax4.text(6, 3.0, 'Y = H × X + Noise', fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax4.text(6, 2.0, 'Goal: Estimate H from Y', fontsize=10, fontweight='bold', ha='center')
ax4.text(6, 1.5, '(knowing X at pilot positions)', fontsize=9, ha='center', style='italic')

ax4.text(6, 0.5, 'NMSE = ||Ĥ - H||² / ||H||²', fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# ==================== Subplot 5: Your Results ====================
ax5 = fig.add_subplot(2, 3, 5)

snr_vals = np.array([5, 10, 15, 20, 25, 30])
pilot_only = np.array([-5.79, -10.22, -13.81, -16.06, -17.11, -17.51])
full_pipeline = np.array([-13.70, -15.56, -16.63, -17.43, -17.48, -17.52])

width = 1.5
x = snr_vals

bars1 = ax5.bar(x - width/2, pilot_only, width, label='Pilot Only', 
               color='lightcoral', edgecolor='black', linewidth=1.5)
bars2 = ax5.bar(x + width/2, full_pipeline, width, label='Full Pipeline (DD+Diffusion)',
               color='lightgreen', edgecolor='black', linewidth=1.5)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height - 0.5,
            f'{height:.1f}', ha='center', va='top', fontsize=8, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height - 0.5,
            f'{height:.1f}', ha='center', va='top', fontsize=8, fontweight='bold')

ax5.set_xlabel('Input SNR (dB)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Output NMSE (dB)', fontsize=12, fontweight='bold')
ax5.set_title('Your Results: Input SNR vs Output NMSE', fontsize=14, fontweight='bold')
ax5.legend(fontsize=10, loc='upper left')
ax5.grid(True, alpha=0.3, axis='y')
ax5.set_xticks(snr_vals)

# Add annotation
ax5.text(17.5, -5, 'Lower NMSE = Better Estimation\n(less error)', 
        fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# ==================== Subplot 6: Improvement ====================
ax6 = fig.add_subplot(2, 3, 6)

improvement = full_pipeline - pilot_only

bars = ax6.bar(snr_vals, improvement, width=2, color='mediumseagreen', 
              edgecolor='black', linewidth=1.5)

# Add value labels
for i, (snr, imp) in enumerate(zip(snr_vals, improvement)):
    ax6.text(snr, imp + 0.3, f'{imp:.2f} dB', ha='center', fontsize=9, fontweight='bold')

ax6.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax6.set_xlabel('Input SNR (dB)', fontsize=12, fontweight='bold')
ax6.set_ylabel('NMSE Improvement (dB)', fontsize=12, fontweight='bold')
ax6.set_title('Gain: Full Pipeline vs Baseline', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')
ax6.set_xticks(snr_vals)

# Highlight best gains
ax6.text(10, 6, 'Best Gains at\nLow-Medium SNR!', fontsize=11, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig('/home/ahmed/selective-decision-feedback/results/SNR_NMSE_EXPLANATION.png', dpi=150, bbox_inches='tight')
print("Diagram saved to: results/SNR_NMSE_EXPLANATION.png")
