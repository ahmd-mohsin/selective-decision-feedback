import sys
from pathlib import Path

script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import yaml
import numpy as np
from transmission_system.config import TransmissionConfig
from transmission_system.dataset_generator import (
    generate_and_save_all_splits, 
    load_dataset_hdf5, 
    verify_dataset_shapes
)


def create_default_config():
    return {
        'seed': 42,
        'Nfft': 64,
        'Ncp': 16,
        'Nsym': 100,
        'pilot_spacing': 4,
        'Nt': 1,
        'Nr': 1,
        'modulation_order': 16,
        'constellation_type': 'qam',
        'snr_db': 20.0,
        'channel_model': 'rayleigh',
        'num_taps': 6,
        'doppler_hz': 100.0,
        'time_varying': True,
        'block_fading': False,
        'num_samples_train': 10000,
        'num_samples_val': 1000,
        'num_samples_test': 1000,
        'dataset_path': './datasets'
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate OFDM datasets for diffusion model training (Module 2)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration YAML file')
    
    parser.add_argument('--Nfft', type=int, default=64,
                       help='FFT size')
    parser.add_argument('--Ncp', type=int, default=16,
                       help='Cyclic prefix length')
    parser.add_argument('--Nsym', type=int, default=100,
                       help='Number of OFDM symbols per sample')
    parser.add_argument('--pilot_spacing', type=int, default=4,
                       help='Pilot spacing (every N subcarriers)')
    
    parser.add_argument('--Nt', type=int, default=1,
                       help='Number of transmit antennas')
    parser.add_argument('--Nr', type=int, default=1,
                       help='Number of receive antennas')
    
    parser.add_argument('--modulation_order', type=int, default=16,
                       choices=[4, 16, 64, 256],
                       help='Modulation order (4, 16, 64, 256)')
    parser.add_argument('--constellation_type', type=str, default='qam',
                       choices=['qam', 'psk'],
                       help='Constellation type')
    
    parser.add_argument('--snr_db', type=float, default=20.0,
                       help='SNR in dB')
    parser.add_argument('--snr_range', type=str, default=None,
                       help='SNR range for multi-SNR dataset (e.g., "10,15,20,25")')
    
    parser.add_argument('--channel_model', type=str, default='rayleigh',
                       choices=['rayleigh', 'rician', 'tapped_delay'],
                       help='Channel model')
    parser.add_argument('--num_taps', type=int, default=6,
                       help='Number of channel taps')
    
    parser.add_argument('--doppler_hz', type=float, default=100.0,
                       help='Doppler frequency in Hz')
    parser.add_argument('--doppler_range', type=str, default=None,
                       help='Doppler range for multi-Doppler dataset (e.g., "50,100,150,200")')
    
    parser.add_argument('--time_varying', action='store_true', default=True,
                       help='Enable time-varying channel')
    parser.add_argument('--block_fading', action='store_true',
                       help='Enable block fading (static per frame)')
    
    parser.add_argument('--num_train', type=int, default=10000,
                       help='Number of training samples')
    parser.add_argument('--num_val', type=int, default=1000,
                       help='Number of validation samples')
    parser.add_argument('--num_test', type=int, default=1000,
                       help='Number of test samples')
    
    parser.add_argument('--dataset_path', type=str, default='./datasets',
                       help='Path to save datasets')
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='Custom dataset name (auto-generated if not provided)')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    parser.add_argument('--multi_condition', action='store_true',
                       help='Generate multi-condition dataset (varying SNR and Doppler)')
    
    args = parser.parse_args()
    
    if args.config is not None:
        print(f"Loading configuration from: {args.config}")
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = TransmissionConfig(**config_dict)
    else:
        config = TransmissionConfig(
            seed=args.seed,
            Nfft=args.Nfft,
            Ncp=args.Ncp,
            Nsym=args.Nsym,
            pilot_spacing=args.pilot_spacing,
            Nt=args.Nt,
            Nr=args.Nr,
            modulation_order=args.modulation_order,
            constellation_type=args.constellation_type,
            snr_db=args.snr_db,
            channel_model=args.channel_model,
            num_taps=args.num_taps,
            doppler_hz=args.doppler_hz,
            time_varying=args.time_varying,
            block_fading=args.block_fading,
            num_samples_train=args.num_train,
            num_samples_val=args.num_val,
            num_samples_test=args.num_test,
            dataset_path=args.dataset_path
        )
    
    print("\n" + "=" * 70)
    print("DATASET GENERATION FOR MODULE 2 (DIFFUSION TRAINING)")
    print("=" * 70)
    
    print("\nSystem Configuration:")
    print(f"  FFT Size (Nfft):          {config.Nfft}")
    print(f"  CP Length (Ncp):          {config.Ncp}")
    print(f"  Symbols per Sample:       {config.Nsym}")
    print(f"  Pilot Spacing:            Every {config.pilot_spacing} subcarriers")
    print(f"  Data Tones:               {config.num_data_tones} per symbol")
    print(f"  Pilot Tones:              {config.Nfft // config.pilot_spacing} per symbol")
    
    print("\nMIMO Configuration:")
    print(f"  Transmit Antennas (Nt):   {config.Nt}")
    print(f"  Receive Antennas (Nr):    {config.Nr}")
    print(f"  MIMO Mode:                {'SISO' if config.Nt == 1 and config.Nr == 1 else f'{config.Nt}x{config.Nr} MIMO'}")
    
    print("\nModulation:")
    print(f"  Modulation Order:         {config.modulation_order}-{config.constellation_type.upper()}")
    print(f"  Bits per Symbol:          {config.bits_per_symbol}")
    
    print("\nChannel Configuration:")
    print(f"  Channel Model:            {config.channel_model}")
    print(f"  Number of Taps:           {config.num_taps}")
    print(f"  SNR:                      {config.snr_db} dB")
    print(f"  Doppler Frequency:        {config.doppler_hz} Hz")
    print(f"  Normalized Doppler:       {config.max_doppler_norm:.6f}")
    print(f"  Time Varying:             {config.time_varying}")
    print(f"  Block Fading:             {config.block_fading}")
    
    print("\nDataset Sizes:")
    print(f"  Training Samples:         {config.num_samples_train:,}")
    print(f"  Validation Samples:       {config.num_samples_val:,}")
    print(f"  Test Samples:             {config.num_samples_test:,}")
    print(f"  Total Samples:            {config.num_samples_train + config.num_samples_val + config.num_samples_test:,}")
    
    print("\nOutput Configuration:")
    print(f"  Dataset Path:             {config.dataset_path}")
    print(f"  Random Seed:              {config.seed}")
    
    print("\n" + "=" * 70)
    print("EXPECTED OUTPUT SHAPES FOR MODULE 2 (DIFFUSION)")
    print("=" * 70)
    
    if config.Nt == 1 and config.Nr == 1:
        print("\nSISO Mode Tensors:")
        print(f"  Y_grid:        [{config.Nsym}, {config.Nfft}] complex")
        print(f"  H_true:        [{config.Nsym}, {config.Nfft}] complex  <- Ground truth for supervision")
        print(f"  H_pilot_full:  [{config.Nsym}, {config.Nfft}] complex  <- Conditioning (coarse)")
        print(f"  pilot_mask:    [{config.Nsym}, {config.Nfft}] bool     <- Conditioning")
        print(f"  Yp:            [num_pilots] complex                    <- Conditioning")
        print(f"  Xp:            [num_pilots] complex                    <- Conditioning")
        print(f"  noise_var:     scalar                                  <- Conditioning")
    else:
        print("\nMIMO Mode Tensors:")
        print(f"  Y_grid:        [{config.Nsym}, {config.Nfft}, {config.Nr}] complex")
        print(f"  H_true:        [{config.Nsym}, {config.Nfft}, {config.Nr}, {config.Nt}] complex  <- Supervision")
        print(f"  H_pilot_full:  [{config.Nsym}, {config.Nfft}, {config.Nr}, {config.Nt}] complex  <- Conditioning")
        print(f"  pilot_mask:    [{config.Nsym}, {config.Nfft}] bool")
        print(f"  Yp:            [num_pilots, {config.Nr}] complex")
        print(f"  Xp:            [num_pilots] complex")
        print(f"  noise_var:     scalar")
    
    print("\nNote for Module 2 Implementation:")
    print("  - Convert complex tensors to [Re, Im] channels for neural networks")
    print("  - Use H_true as supervision target")
    print("  - Use H_pilot_full + pilot_mask as primary conditioning")
    print("  - Optionally include dd_mask + (X_dd, Y_dd) for Mode B training")
    
    print("\n" + "=" * 70)
    
    response = input("\nProceed with dataset generation? [y/N]: ")
    if response.lower() != 'y':
        print("Dataset generation cancelled.")
        return
    
    print("\n" + "=" * 70)
    print("GENERATING DATASETS...")
    print("=" * 70 + "\n")
    
    if args.multi_condition or args.snr_range or args.doppler_range:
        print("Multi-condition dataset generation not yet implemented.")
        print("Generating single-condition dataset instead.\n")
    
    dataset_paths = generate_and_save_all_splits(config)
    
    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 70)
    
    print("\nGenerated Files:")
    for split, path in dataset_paths.items():
        file_size = Path(path).stat().st_size / (1024 * 1024)
        print(f"  {split:10s}: {path}")
        print(f"              Size: {file_size:.2f} MB")
    
    print("\nVerifying Training Dataset...")
    train_dataset, train_config = load_dataset_hdf5(dataset_paths['train'])
    verify_dataset_shapes(train_dataset, config)
    
    print("\n" + "=" * 70)
    print("QUICK START GUIDE FOR MODULE 2")
    print("=" * 70)
    
    print("\nLoading Dataset in Python:")
    print("```python")
    print("from transmission_system.dataset_generator import OFDMDataset")
    print()
    print(f"train_dataset = OFDMDataset('{dataset_paths['train']}')")
    print(f"val_dataset = OFDMDataset('{dataset_paths['val']}')")
    print()
    print("# Get single sample")
    print("sample = train_dataset[0]")
    print()
    print("# Get conditioning for diffusion")
    print("conditioning = train_dataset.get_diffusion_conditioning(0)")
    print("ground_truth = train_dataset.get_ground_truth(0)")
    print()
    print("# Batch loading")
    print("indices = np.arange(32)  # batch size 32")
    print("batch = train_dataset.get_batch(indices)")
    print("```")
    
    print("\nDiffusion Training Pseudocode:")
    print("```python")
    print("for epoch in range(num_epochs):")
    print("    for batch_idx in dataloader:")
    print("        # Get data")
    print("        H_true = batch['H_true']  # Ground truth")
    print("        H_pilot = batch['H_pilot_full']  # Conditioning")
    print("        pilot_mask = batch['pilot_mask']")
    print("        ")
    print("        # Convert complex to [Re, Im] channels")
    print("        H_true_re_im = complex_to_channels(H_true)")
    print("        H_pilot_re_im = complex_to_channels(H_pilot)")
    print("        ")
    print("        # Forward diffusion")
    print("        t = sample_timesteps(batch_size)")
    print("        noise = randn_like(H_true_re_im)")
    print("        H_noisy = q_sample(H_true_re_im, t, noise)")
    print("        ")
    print("        # Predict noise")
    print("        cond = concat([H_pilot_re_im, pilot_mask], dim=1)")
    print("        noise_pred = model(H_noisy, t, cond)")
    print("        ")
    print("        # Loss")
    print("        loss = mse_loss(noise_pred, noise)")
    print("        loss.backward()")
    print("```")
    
    print("\n" + "=" * 70)
    print(f"Dataset ready for Module 2 training!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()