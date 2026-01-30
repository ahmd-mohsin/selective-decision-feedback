# main_generate_dataset.py

import numpy as np
from transmission import SystemConfig, DatasetGenerator

def main():
    config = SystemConfig(
        Nr=16,
        Nt=64,
        Np=38,
        num_data_symbols=100,
        modulation_order=16,
        snr_db=10.0,
        channel_model='CDL-C',
        num_samples=10000
    )
    
    generator = DatasetGenerator(config)
    
    print("Generating training dataset...")
    train_dataset = generator.generate_dataset(num_samples=8000)
    generator.save_dataset(train_dataset, 'data/train_dataset.npy')
    
    print("\nGenerating validation dataset...")
    val_dataset = generator.generate_dataset(num_samples=1000)
    generator.save_dataset(val_dataset, 'data/val_dataset.npy')
    
    print("\nGenerating test dataset...")
    test_dataset = generator.generate_dataset(num_samples=1000)
    generator.save_dataset(test_dataset, 'data/test_dataset.npy')
    
    print("\nDataset generation complete!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

if __name__ == "__main__":
    main()