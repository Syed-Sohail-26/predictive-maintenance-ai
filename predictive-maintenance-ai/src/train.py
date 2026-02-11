"""
Training Script for Predictive Maintenance Models
Author: Syed Sohail Ahmed
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.lstm_model import LSTMPredictiveModel
from models.random_forest_model import RandomForestPredictiveModel
from src.data_preprocessing import DataPreprocessor


def train_lstm(args):
    """
    Train LSTM model for predictive maintenance.
    """
    print("\n" + "="*60)
    print("Training LSTM Model for Predictive Maintenance")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(scaling_method='standard')
    
    # Load and preprocess data
    print("\n[1/5] Loading data...")
    df = preprocessor.load_data(args.data_path)
    
    print("\n[2/5] Preprocessing data...")
    df = preprocessor.handle_missing_values(df, strategy='mean')
    df = preprocessor.create_time_features(df)
    
    # Create sequences
    print("\n[3/5] Creating sequences...")
    X, y = preprocessor.create_sequences(df, sequence_length=args.sequence_length)
    
    # Split data
    print("\n[4/5] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        X, y, test_size=0.2, val_size=0.1
    )
    
    # Initialize and train model
    print("\n[5/5] Training LSTM model...")
    model = LSTMPredictiveModel(
        sequence_length=args.sequence_length,
        num_features=X.shape[2]
    )
    
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    metrics = model.evaluate(X_test, y_test)
    
    print("\n=== Final Results ===")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # Save model
    model_path = f"models/lstm_model_final.h5"
    model.save_model(model_path)
    
    print(f"\n✓ Training completed successfully!")
    print(f"✓ Model saved to {model_path}")
    

def train_random_forest(args):
    """
    Train Random Forest model for predictive maintenance.
    """
    print("\n" + "="*60)
    print("Training Random Forest Model for Predictive Maintenance")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(scaling_method='standard')
    
    # Load and preprocess data
    print("\n[1/4] Loading data...")
    df = preprocessor.load_data(args.data_path)
    
    print("\n[2/4] Preprocessing data...")
    df = preprocessor.handle_missing_values(df, strategy='mean')
    df = preprocessor.create_time_features(df)
    
    # Prepare features and target
    target_col = 'failure'
    feature_cols = [col for col in df.columns if col not in [target_col, 'timestamp']]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Scale features
    print("\n[3/4] Scaling features...")
    X_train_temp, X_test, y_train_temp, y_test = preprocessor.split_data(
        X, y, test_size=0.2, val_size=0
    )[:4]
    
    X_train, X_test = preprocessor.scale_features(X_train_temp, X_test)
    y_train = y_train_temp
    
    # Initialize and train model
    print("\n[4/4] Training Random Forest model...")
    model = RandomForestPredictiveModel(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    
    model.train(X_train, y_train, feature_names=feature_cols)
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    metrics = model.evaluate(X_test, y_test)
    
    # Show feature importance
    print("\nTop 10 Most Important Features:")
    print(model.get_feature_importance(10))
    
    # Save model
    model_path = f"models/random_forest_model_final.pkl"
    model.save_model(model_path)
    
    print(f"\n✓ Training completed successfully!")
    print(f"✓ Model saved to {model_path}")


def main():
    """
    Main training function.
    """
    parser = argparse.ArgumentParser(
        description='Train Predictive Maintenance Models'
    )
    
    # Common arguments
    parser.add_argument(
        '--model',
        type=str,
        choices=['lstm', 'rf'],
        required=True,
        help='Model type: lstm or rf (random forest)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/sample_sensor_data.csv',
        help='Path to training data CSV file'
    )
    
    # LSTM-specific arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (LSTM only)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (LSTM only)'
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=24,
        help='Sequence length for LSTM'
    )
    
    # Random Forest-specific arguments
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Number of trees in Random Forest'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=20,
        help='Maximum depth of trees in Random Forest'
    )
    
    args = parser.parse_args()
    
    # Train the selected model
    if args.model == 'lstm':
        train_lstm(args)
    elif args.model == 'rf':
        train_random_forest(args)


if __name__ == "__main__":
    main()
