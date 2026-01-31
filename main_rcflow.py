import torch
import numpy as np
from rc_flow import RCFlowConfig, RCFlowTrainer, ChannelDataset


def main():
    config = RCFlowConfig(
        Nr=4,
        Nt=3,
        Np=2,
        snr_db=10.0,
        hidden_dim=128,
        num_layers=4,
        num_flow_steps=50,
        num_outer_iterations=3,
        learning_rate=1e-3,
        batch_size=32,
        num_epochs=100
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
        num_train=5000,
        num_val=500,
        num_test=500
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
        num_epochs=config.num_epochs
    )

    print("\n" + "=" * 50)
    print("Evaluation on Test Set")
    print("=" * 50)

    test_nmse = trainer.evaluate(test_data['H_true'], test_data['pilot_mask'])
    print(f"Test NMSE: {test_nmse:.2f} dB")

    print("\nComparing with baselines...")
    ls_nmse = compute_ls_baseline(test_data, config)
    print(f"LS Estimation NMSE: {ls_nmse:.2f} dB")

    oracle_nmse = compute_oracle_baseline(test_data, config)
    print(f"Oracle (Perfect CSI) NMSE: {oracle_nmse:.2f} dB")

    print("\n" + "=" * 50)
    print("Reconstruction Example")
    print("=" * 50)

    idx = 0
    H_true = test_data['H_true'][idx:idx+1]
    H_noisy = test_data['H_noisy'][idx:idx+1]
    pilot_mask = test_data['pilot_mask'][idx:idx+1]

    H_reconstructed = trainer.reconstruct(H_noisy, pilot_mask)

    print(f"H_true (sample):\n{H_true[0, :2, :].numpy()}")
    print(f"\nH_noisy (sample):\n{H_noisy[0, :2, :].numpy()}")
    print(f"\nH_reconstructed (sample):\n{H_reconstructed[0, :2, :].cpu().numpy()}")

    sample_nmse = trainer.compute_nmse(H_reconstructed, H_true.to(config.device))
    print(f"\nSample NMSE: {sample_nmse:.2f} dB")

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


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
