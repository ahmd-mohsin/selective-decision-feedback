import torch
from rc_flow import RCFlowConfig, RCFlowTrainer, ChannelDataset


def main():
    config = RCFlowConfig(
        Nr=4,
        Nt=3,
        Np=6,
        snr_db=10.0,
        hidden_dim=512,
        num_layers=12,
        num_flow_steps=100,
        learning_rate=1e-3,
        batch_size=64,
        num_epochs=300
    )

    print("=" * 60)
    print("RC-Flow Channel Estimation (Heavy Network)")
    print("=" * 60)
    print(f"Channel: {config.Nr} x {config.Nt} = {config.total_elements} elements")
    print(f"Pilots: {config.Np} ({config.pilot_density:.1%} density)")
    print(f"SNR: {config.snr_db} dB")
    print(f"Network: {config.hidden_dim} hidden, {config.num_layers} layers")
    print(f"Device: {config.device}")
    print("=" * 60)

    print("\n[1] Generating dataset...")
    dataset = ChannelDataset(config, num_angles=8)
    train_data, val_data, test_data = dataset.generate_train_val_test(
        num_train=20000,
        num_val=2000,
        num_test=2000
    )
    print(f"    Train: {train_data['H_true'].shape[0]}")
    print(f"    Val:   {val_data['H_true'].shape[0]}")
    print(f"    Test:  {test_data['H_true'].shape[0]}")

    print("\n[2] Initializing trainer...")
    trainer = RCFlowTrainer(config)

    num_params = sum(p.numel() for p in trainer.flow.parameters())
    print(f"    Model parameters: {num_params:,}")

    print("\n[3] Training...")
    print("-" * 60)
    history = trainer.train(
        train_data['H_true'],
        train_data['pilot_mask'],
        val_data['H_true'],
        val_data['pilot_mask'],
        num_epochs=config.num_epochs
    )

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    H_true = test_data['H_true']
    H_noisy = test_data['H_noisy']
    pilot_mask = test_data['pilot_mask']

    H_observed = torch.zeros_like(H_noisy)
    H_observed[pilot_mask] = H_noisy[pilot_mask]

    print("\n[4] Baselines:")
    oracle = compute_nmse(H_noisy, H_true)
    print(f"    Oracle (full noisy):  {oracle:.2f} dB")

    zero_fill = compute_nmse(H_observed, H_true)
    print(f"    Zero-fill:            {zero_fill:.2f} dB")

    mean_fill = compute_mean_fill(H_observed, pilot_mask)
    mean_nmse = compute_nmse(mean_fill, H_true)
    print(f"    Mean-fill:            {mean_nmse:.2f} dB")

    print("\n[5] RC-Flow:")
    H_rcflow = trainer.reconstruct(H_observed, pilot_mask)
    rcflow_nmse = compute_nmse(H_rcflow.cpu(), H_true)
    print(f"    RC-Flow:              {rcflow_nmse:.2f} dB")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"    Oracle:     {oracle:.2f} dB")
    print(f"    RC-Flow:    {rcflow_nmse:.2f} dB")
    print(f"    Mean-fill:  {mean_nmse:.2f} dB")
    print(f"    Zero-fill:  {zero_fill:.2f} dB")

    if rcflow_nmse < mean_nmse:
        print("\n    >>> RC-Flow BEATS mean-fill! <<<")

    print("\n[6] Saving...")
    trainer.save("results/rcflow_model.pt")
    print("    Done: results/rcflow_model.pt")


def compute_nmse(H_est: torch.Tensor, H_true: torch.Tensor) -> float:
    mse = torch.mean(torch.abs(H_est - H_true) ** 2)
    power = torch.mean(torch.abs(H_true) ** 2)
    return 10 * torch.log10(mse / power).item()


def compute_mean_fill(H_observed: torch.Tensor, pilot_mask: torch.Tensor) -> torch.Tensor:
    H_filled = H_observed.clone()
    for b in range(H_observed.shape[0]):
        obs = H_observed[b][pilot_mask[b]]
        mean_val = obs.mean() if len(obs) > 0 else 0
        H_filled[b] = torch.where(pilot_mask[b], H_observed[b], mean_val)
    return H_filled


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
