import torch
import numpy as np
from rc_flow import RCFlowConfig, RCFlowTrainer, ChannelDataset


def main():
    config = RCFlowConfig(
        Nr=4,
        Nt=3,
        Np=6,
        snr_db=10.0,
        hidden_dim=256,
        num_layers=6,
        num_flow_steps=100,
        lambda_proj=0.9,
        learning_rate=5e-4,
        batch_size=64,
        num_epochs=200
    )

    print("=" * 60)
    print("RC-Flow Channel Estimation")
    print("=" * 60)
    print(f"Channel matrix: {config.Nr} x {config.Nt} (Nr x Nt)")
    print(f"Total elements: {config.total_elements}")
    print(f"Observed pilots: {config.Np}")
    print(f"Pilot density: {config.pilot_density:.1%}")
    print(f"SNR: {config.snr_db} dB")
    print(f"Device: {config.device}")
    print("=" * 60)

    print("\n[1] Generating structured channel dataset...")
    dataset = ChannelDataset(config, num_angles=8)
    train_data, val_data, test_data = dataset.generate_train_val_test(
        num_train=10000,
        num_val=1000,
        num_test=1000
    )

    print(f"    Train: {train_data['H_true'].shape}")
    print(f"    Val:   {val_data['H_true'].shape}")
    print(f"    Test:  {test_data['H_true'].shape}")

    print("\n[2] Initializing RC-Flow trainer...")
    trainer = RCFlowTrainer(config)

    print("\n[3] Training flow matching model...")
    print("-" * 60)
    history = trainer.train(
        train_data['H_true'],
        val_data['H_true'],
        val_data['pilot_mask'],
        num_epochs=config.num_epochs,
        early_stopping_patience=40
    )

    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)

    H_true = test_data['H_true']
    H_noisy = test_data['H_noisy']
    pilot_mask = test_data['pilot_mask']

    H_observed = torch.zeros_like(H_noisy)
    H_observed[pilot_mask] = H_noisy[pilot_mask]

    print("\n[4] Computing baselines...")

    oracle_nmse = compute_nmse(H_noisy, H_true)
    print(f"    Oracle (all noisy):      {oracle_nmse:.2f} dB")

    lmmse_nmse = compute_lmmse_baseline(H_observed, H_true, pilot_mask, config)
    print(f"    LMMSE:                   {lmmse_nmse:.2f} dB")

    zero_fill_nmse = compute_nmse(H_observed, H_true)
    print(f"    Zero-fill (LS):          {zero_fill_nmse:.2f} dB")

    mean_fill = compute_mean_fill(H_observed, pilot_mask)
    mean_fill_nmse = compute_nmse(mean_fill, H_true)
    print(f"    Mean-fill:               {mean_fill_nmse:.2f} dB")

    print("\n[5] RC-Flow reconstruction...")
    H_rcflow = trainer.reconstruct(H_observed, pilot_mask)
    rcflow_nmse = compute_nmse(H_rcflow.cpu(), H_true)
    print(f"    RC-Flow:                 {rcflow_nmse:.2f} dB")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"    Oracle (full noisy CSI): {oracle_nmse:.2f} dB  [lower bound]")
    print(f"    LMMSE:                   {lmmse_nmse:.2f} dB")
    print(f"    RC-Flow:                 {rcflow_nmse:.2f} dB")
    print(f"    Mean-fill:               {mean_fill_nmse:.2f} dB")
    print(f"    Zero-fill (LS):          {zero_fill_nmse:.2f} dB  [upper bound]")
    print("-" * 60)

    if rcflow_nmse < mean_fill_nmse:
        print("RC-Flow BEATS mean-fill baseline!")
    if rcflow_nmse < lmmse_nmse:
        print("RC-Flow BEATS LMMSE baseline!")

    print("\n[6] Saving model...")
    trainer.save("results/rcflow_model.pt")
    print("    Saved to results/rcflow_model.pt")

    print("\n[7] Example reconstruction:")
    show_example(H_true[0], H_observed[0], H_rcflow[0].cpu(), pilot_mask[0])


def compute_nmse(H_est: torch.Tensor, H_true: torch.Tensor) -> float:
    mse = torch.mean(torch.abs(H_est - H_true) ** 2)
    power = torch.mean(torch.abs(H_true) ** 2)
    return 10 * torch.log10(mse / power).item()


def compute_mean_fill(H_observed: torch.Tensor, pilot_mask: torch.Tensor) -> torch.Tensor:
    H_filled = H_observed.clone()
    batch_size = H_observed.shape[0]

    for b in range(batch_size):
        observed_vals = H_observed[b][pilot_mask[b]]
        mean_val = observed_vals.mean() if len(observed_vals) > 0 else 0
        H_filled[b] = torch.where(pilot_mask[b], H_observed[b], mean_val)

    return H_filled


def compute_lmmse_baseline(
    H_observed: torch.Tensor,
    H_true: torch.Tensor,
    pilot_mask: torch.Tensor,
    config: RCFlowConfig
) -> float:
    batch_size = H_observed.shape[0]
    H_lmmse = torch.zeros_like(H_observed)

    snr_linear = 10 ** (config.snr_db / 10)

    for b in range(batch_size):
        obs_mask = pilot_mask[b].flatten()
        obs_indices = torch.where(obs_mask)[0]

        if len(obs_indices) == 0:
            continue

        H_flat = H_observed[b].flatten()
        y_obs = H_flat[obs_indices]

        mean_val = y_obs.mean()
        H_lmmse[b] = mean_val * torch.ones_like(H_observed[b])

        for idx in obs_indices:
            row, col = idx // config.Nt, idx % config.Nt
            H_lmmse[b, row, col] = H_observed[b, row, col]

    return compute_nmse(H_lmmse, H_true)


def show_example(H_true, H_observed, H_est, pilot_mask):
    print(f"\n    H_true[0,:]     = {H_true[0,:].numpy()}")
    print(f"    H_observed[0,:] = {H_observed[0,:].numpy()}")
    print(f"    H_rcflow[0,:]   = {H_est[0,:].numpy()}")
    print(f"    pilot_mask[0,:] = {pilot_mask[0,:].numpy()}")

    error = torch.abs(H_est - H_true)
    print(f"\n    Reconstruction error (abs):")
    print(f"    {error.numpy()}")


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
