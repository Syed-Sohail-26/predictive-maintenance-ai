"""
Data Preprocessing for Predictive Maintenance
Author: Syed Sohail Ahmed
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    Preprocesses sensor data for predictive maintenance models.
    """
    
    def __init__(self, scaling_method='standard'):
        """
        Initialize the preprocessor.
        
        Args:
            scaling_method (str): 'standard' or 'minmax'
        """
        self.scaler = StandardScaler() if scaling_method == 'standard' else MinMaxScaler()
        self.feature_columns = None
    
    def load_data(self, filepath):
        """
        Load sensor data from CSV file.
        
        Args:
            filepath (str): Path to CSV file
            
        Returns:
            DataFrame with sensor data
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        print(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def handle_missing_values(self, df, strategy='mean'):
        """
        Handle missing values in the dataset.
        
        Args:
            df (DataFrame): Input data
            strategy (str): 'mean', 'median', or 'forward_fill'
            
        Returns:
            DataFrame with handled missing values
        """
        print("Handling missing values...")
        
        if strategy == 'mean':
            df = df.fillna(df.mean())
        elif strategy == 'median':
            df = df.fillna(df.median())
        elif strategy == 'forward_fill':
            df = df.fillna(method='ffill')
        
        return df
    
    def remove_outliers(self, df, columns, threshold=3):
        """
        Remove outliers using z-score method.
        
        Args:
            df (DataFrame): Input data
            columns (list): Columns to check for outliers
            threshold (float): Z-score threshold
            
        Returns:
            DataFrame with outliers removed
        """
        print(f"Removing outliers (z-score > {threshold})...")
        
        for col in columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores < threshold]
        
        print(f"Remaining records: {len(df)}")
        return df
    
    def create_time_features(self, df, timestamp_col='timestamp'):
        """
        Create time-based features from timestamp.
        
        Args:
            df (DataFrame): Input data
            timestamp_col (str): Name of timestamp column
            
        Returns:
            DataFrame with additional time features
        """
        if timestamp_col not in df.columns:
            return df
        
        print("Creating time-based features...")
        
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['day_of_month'] = df[timestamp_col].dt.day
        df['month'] = df[timestamp_col].dt.month
        
        # Cyclical encoding for hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        return df
    
    def scale_features(self, X_train, X_test=None):
        """
        Scale features using the specified scaler.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            
        Returns:
            Scaled features
        """
        print("Scaling features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def create_sequences(self, data, sequence_length=24, target_col='failure'):
        """
        Create sequences for LSTM model.
        
        Args:
            data (DataFrame): Input data
            sequence_length (int): Length of each sequence
            target_col (str): Name of target column
            
        Returns:
            X (sequences), y (labels)
        """
        print(f"Creating sequences of length {sequence_length}...")
        
        # Separate features and target
        feature_cols = [col for col in data.columns if col != target_col and col != 'timestamp']
        self.feature_columns = feature_cols
        
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            # Get sequence of features
            sequence = data[feature_cols].iloc[i:i+sequence_length].values
            # Get target (failure at end of sequence)
            label = data[target_col].iloc[i+sequence_length]
            
            X.append(sequence)
            y.append(label)
        
        print(f"Created {len(X)} sequences")
        return np.array(X), np.array(y)
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Features
            y: Labels
            test_size (float): Proportion of test data
            val_size (float): Proportion of validation data
            random_state (int): Random seed
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("Splitting data into train/val/test sets...")
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_statistics(self, df):
        """
        Get basic statistics of the dataset.
        
        Args:
            df (DataFrame): Input data
        """
        print("\n=== Dataset Statistics ===")
        print(f"Total records: {len(df)}")
        print(f"Total features: {len(df.columns)}")
        print(f"\nMissing values:")
        print(df.isnull().sum())
        print(f"\nData types:")
        print(df.dtypes)
        print(f"\nBasic statistics:")
        print(df.describe())


if __name__ == "__main__":
    # Example usage
    print("Data Preprocessing Module")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(scaling_method='standard')
    
    # Create sample data
    print("\nCreating sample sensor data...")
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H'),
        'temperature': np.random.normal(75, 10, 1000),
        'vibration': np.random.normal(0.15, 0.05, 1000),
        'pressure': np.random.normal(101, 5, 1000),
        'rpm': np.random.normal(1500, 100, 1000),
        'failure': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
    })
    
    # Get statistics
    preprocessor.get_statistics(sample_data)
    
    # Create time features
    sample_data = preprocessor.create_time_features(sample_data)
    
    # Create sequences
    X, y = preprocessor.create_sequences(sample_data, sequence_length=24)
    
    print(f"\nSequence shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print("\nPreprocessing complete!")
