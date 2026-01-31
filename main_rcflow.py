import torch
import numpy as np
from rc_flow import RCFlowConfig, RCFlowTrainer, ChannelDataset


def main():
    config = RCFlowConfig(
        Nr=4,
        Nt=3,
        Np=2,
        snr_db=10.0,
        hidden_dim=256,
        num_layers=6,
        num_flow_steps=100,
        num_outer_iterations=5,
        learning_rate=5e-4,
        batch_size=64,
        num_epochs=200,
        lambda_proj=0.8,
        beta_anchor=0.2
    )

    print("=" * 50)
    print("RC-Flow Channel Estimation")
    print("=" * 50)
    print(f"Nr (Rx antennas): {config.Nr}")
    print(f"Nt (Users/Tx): {config.Nt}")
    print(f"Np (Pilots): {config.Np}")
    print(f"Pilot density: {config.pilot_density:.1%}")
    print(f"SNR: {config.snr_db} dB")
    print(f"Device: {config.device}")
    print("=" * 50)

    print("\nGenerating dataset...")
    dataset = ChannelDataset(config)
    train_data, val_data, test_data = dataset.generate_train_val_test(
        num_train=10000,
        num_val=1000,
        num_test=1000
    )

    print(f"Train samples: {train_data['H_true'].shape[0]}")
    print(f"Val samples: {val_data['H_true'].shape[0]}")
    print(f"Test samples: {test_data['H_true'].shape[0]}")
    print(f"Channel shape: {train_data['H_true'].shape[1:]}")

    print("\nInitializing trainer...")
    trainer = RCFlowTrainer(config)

    print("\nTraining flow matching model...")
    history = trainer.train(
        train_data['H_true'],
        val_data['H_true'],
        num_epochs=config.num_epochs,
        early_stopping_patience=30
    )

    print("\n" + "=" * 50)
    print("Evaluation on Test Set")
    print("=" * 50)

    test_nmse = trainer.evaluate(test_data['H_true'], test_data['pilot_mask'])
    print(f"RC-Flow NMSE: {test_nmse:.2f} dB")

    print("\nComparing with baselines...")
    ls_nmse = compute_ls_baseline(test_data, config)
    print(f"LS Estimation NMSE: {ls_nmse:.2f} dB")

    oracle_nmse = compute_oracle_baseline(test_data, config)
    print(f"Oracle (Noisy Full CSI) NMSE: {oracle_nmse:.2f} dB")

    interp_nmse = compute_interpolation_baseline(test_data, config)
    print(f"Linear Interpolation NMSE: {interp_nmse:.2f} dB")

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Oracle (Noisy):     {oracle_nmse:.2f} dB")
    print(f"RC-Flow:            {test_nmse:.2f} dB")
    print(f"Interpolation:      {interp_nmse:.2f} dB")
    print(f"LS (Pilots only):   {ls_nmse:.2f} dB")

    if test_nmse < interp_nmse:
        print("\nRC-Flow outperforms interpolation baseline!")
    else:
        print("\nRC-Flow needs improvement to beat interpolation.")

    print("\nSaving model...")
    trainer.save("results/rcflow_model.pt")
    print("Model saved to results/rcflow_model.pt")


def compute_ls_baseline(data: dict, config: RCFlowConfig) -> float:
    H_true = data['H_true']
    H_noisy = data['H_noisy']
    pilot_mask = data['pilot_mask']

    H_ls = torch.zeros_like(H_noisy)
    mask_expanded = pilot_mask.unsqueeze(1).expand_as(H_noisy)
    H_ls[mask_expanded] = H_noisy[mask_expanded]

    error = torch.abs(H_ls - H_true) ** 2
    power = torch.abs(H_true) ** 2
    nmse = torch.mean(error) / torch.mean(power)
    return 10 * torch.log10(nmse).item()


def compute_oracle_baseline(data: dict, config: RCFlowConfig) -> float:
    H_true = data['H_true']
    H_noisy = data['H_noisy']

    error = torch.abs(H_noisy - H_true) ** 2
    power = torch.abs(H_true) ** 2
    nmse = torch.mean(error) / torch.mean(power)
    return 10 * torch.log10(nmse).item()


def compute_interpolation_baseline(data: dict, config: RCFlowConfig) -> float:
    H_true = data['H_true']
    H_noisy = data['H_noisy']
    pilot_mask = data['pilot_mask']

    batch_size = H_true.shape[0]
    H_interp = torch.zeros_like(H_noisy)

    for b in range(batch_size):
        pilot_indices = torch.where(pilot_mask[b])[0]
        if len(pilot_indices) == 0:
            continue

        for rx in range(config.Nr):
            pilot_values = H_noisy[b, rx, pilot_indices]
            mean_value = pilot_values.mean()
            H_interp[b, rx, :] = mean_value
            H_interp[b, rx, pilot_indices] = pilot_values

    error = torch.abs(H_interp - H_true) ** 2
    power = torch.abs(H_true) ** 2
    nmse = torch.mean(error) / torch.mean(power)
    return 10 * torch.log10(nmse).item()


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
