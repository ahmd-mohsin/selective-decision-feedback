import argparse
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transmission_system.config import TransmissionConfig
from transmission_system.dataset_generator import generate_and_save_all_splits, load_dataset_hdf5, verify_dataset_shapes


def main():
    parser = argparse.ArgumentParser(description='Generate OFDM dataset for diffusion and DD training')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration YAML file')
    parser.add_argument('--Nfft', type=int, default=64,
                       help='FFT size')
    parser.add_argument('--Nsym', type=int, default=100,
                       help='Number of OFDM symbols')
    parser.add_argument('--pilot_spacing', type=int, default=4,
                       help='Pilot spacing (every N subcarriers)')
    parser.add_argument('--modulation_order', type=int, default=16,
                       help='Modulation order (4, 16, 64, 256)')
    parser.add_argument('--snr_db', type=float, default=20.0,
                       help='SNR in dB')
    parser.add_argument('--doppler_hz', type=float, default=100.0,
                       help='Doppler frequency in Hz')
    parser.add_argument('--num_train', type=int, default=10000,
                       help='Number of training samples')
    parser.add_argument('--num_val', type=int, default=1000,
                       help='Number of validation samples')
    parser.add_argument('--num_test', type=int, default=1000,
                       help='Number of test samples')
    parser.add_argument('--dataset_path', type=str, default='./datasets',
                       help='Path to save datasets')
    parser.add_argument('--Nt', type=int, default=1,
                       help='Number of transmit antennas')
    parser.add_argument('--Nr', type=int, default=1,
                       help='Number of receive antennas')
    parser.add_argument('--channel_model', type=str, default='rayleigh',
                       choices=['rayleigh', 'rician', 'tapped_delay'],
                       help='Channel model')
    parser.add_argument('--time_varying', action='store_true',
                       help='Enable time-varying channel')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    if args.config is not None:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = TransmissionConfig(**config_dict)
    else:
        config = TransmissionConfig(
            seed=args.seed,
            Nfft=args.Nfft,
            Nsym=args.Nsym,
            pilot_spacing=args.pilot_spacing,
            modulation_order=args.modulation_order,
            snr_db=args.snr_db,
            doppler_hz=args.doppler_hz,
            num_samples_train=args.num_train,
            num_samples_val=args.num_val,
            num_samples_test=args.num_test,
            dataset_path=args.dataset_path,
            Nt=args.Nt,
            Nr=args.Nr,
            channel_model=args.channel_model,
            time_varying=args.time_varying
        )
    
    print("Dataset Generation Configuration:")
    print("=" * 60)
    print(f"FFT Size (Nfft): {config.Nfft}")
    print(f"CP Length (Ncp): {config.Ncp}")
    print(f"Number of Symbols (Nsym): {config.Nsym}")
    print(f"Pilot Spacing: {config.pilot_spacing}")
    print(f"Modulation: {config.modulation_order}-{config.constellation_type.upper()}")
    print(f"MIMO: {config.Nt}x{config.Nr}")
    print(f"SNR: {config.snr_db} dB")
    print(f"Channel Model: {config.channel_model}")
    print(f"Doppler: {config.doppler_hz} Hz")
    print(f"Time Varying: {config.time_varying}")
    print(f"Block Fading: {config.block_fading}")
    print(f"Training Samples: {config.num_samples_train}")
    print(f"Validation Samples: {config.num_samples_val}")
    print(f"Test Samples: {config.num_samples_test}")
    print(f"Dataset Path: {config.dataset_path}")
    print("=" * 60)
    print()
    
    dataset_paths = generate_and_save_all_splits(config)
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)
    print("\nDataset files:")
    for split, path in dataset_paths.items():
        print(f"  {split}: {path}")
    
    print("\nVerifying training dataset...")
    train_dataset, _ = load_dataset_hdf5(dataset_paths['train'])
    verify_dataset_shapes(train_dataset, config)
    
    print("\n" + "=" * 60)
    print("All datasets ready for diffusion and DD training!")
    print("=" * 60)


if __name__ == '__main__':
    main()