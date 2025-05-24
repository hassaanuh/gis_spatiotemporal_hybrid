#!/usr/bin/env python3
"""
Simple script to run the KSC-ConvLSTM training on your air quality data
"""

import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from train_model import main
    print("Starting KSC-ConvLSTM training on air quality data...")
    print("=" * 60)
    
    # Run the training
    model, history, scaler = main()
    
    print("=" * 60)
    print("Training completed successfully!")
    print("\nFiles created:")
    print("- ksc_convlstm_air_quality_model.h5 (trained model)")
    print("- air_quality_scaler.pkl (data scaler)")
    print("- training_history.png (training plots)")
    
except Exception as e:
    print(f"Error during training: {e}")
    print("\nMake sure you have:")
    print("1. The combined.csv file in grid_data/ directory")
    print("2. All required packages installed (tensorflow, sklearn, pandas, numpy, matplotlib)")
    print("3. Sufficient memory for training")
