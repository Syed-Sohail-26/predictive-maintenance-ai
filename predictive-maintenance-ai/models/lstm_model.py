"""
LSTM Model for Predictive Maintenance
Author: Syed Sohail Ahmed
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class LSTMPredictiveModel:
    """
    LSTM-based model for predicting equipment failures using time-series sensor data.
    """
    
    def __init__(self, sequence_length=24, num_features=5):
        """
        Initialize the LSTM model.
        
        Args:
            sequence_length (int): Number of time steps to look back
            num_features (int): Number of sensor features
        """
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """
        Build the LSTM neural network architecture.
        """
        self.model = Sequential([
            # First LSTM layer with return sequences
            LSTM(128, return_sequences=True, 
                 input_shape=(self.sequence_length, self.num_features)),
            Dropout(0.2),
            
            # Second LSTM layer
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            
            # Dense layers
            Dense(32, activation='relu'),
            Dropout(0.1),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Model checkpoint callback
        checkpoint = ModelCheckpoint(
            'models/best_lstm_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        loss, accuracy, precision, recall = self.model.evaluate(X_test, y_test)
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def get_model_summary(self):
        """
        Print model architecture summary.
        """
        return self.model.summary()


if __name__ == "__main__":
    # Example usage
    print("Building LSTM Predictive Maintenance Model...")
    
    # Initialize model
    model = LSTMPredictiveModel(sequence_length=24, num_features=5)
    
    # Print model summary
    model.get_model_summary()
    
    # Create dummy data for demonstration
    X_train = np.random.randn(1000, 24, 5)
    y_train = np.random.randint(0, 2, 1000)
    X_val = np.random.randn(200, 24, 5)
    y_val = np.random.randint(0, 2, 200)
    
    print("\nModel architecture created successfully!")
    print("Ready for training with real sensor data.")
