"""
Random Forest Model for Predictive Maintenance
Author: Syed Sohail Ahmed
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import joblib
import matplotlib.pyplot as plt


class RandomForestPredictiveModel:
    """
    Random Forest classifier for equipment failure prediction.
    """
    
    def __init__(self, n_estimators=100, max_depth=20, random_state=42):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of trees
            random_state (int): Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,  # Use all available cores
            verbose=1
        )
        self.feature_names = None
        self.feature_importance = None
    
    def train(self, X_train, y_train, feature_names=None):
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: List of feature names
            
        Returns:
            Trained model
        """
        print("Training Random Forest model...")
        self.feature_names = feature_names
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_names if feature_names else [f'feature_{i}' for i in range(X_train.shape[1])],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Training completed!")
        return self.model
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predicted classes
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        # Print detailed results
        print("\n=== Model Evaluation ===")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return metrics
    
    def get_feature_importance(self, top_n=10):
        """
        Get top N most important features.
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            DataFrame of top features
        """
        if self.feature_importance is None:
            print("Model not trained yet!")
            return None
        
        return self.feature_importance.head(top_n)
    
    def plot_feature_importance(self, top_n=10, figsize=(10, 6)):
        """
        Plot feature importance.
        
        Args:
            top_n (int): Number of top features to plot
            figsize (tuple): Figure size
        """
        if self.feature_importance is None:
            print("Model not trained yet!")
            return
        
        plt.figure(figsize=figsize)
        top_features = self.feature_importance.head(top_n)
        
        plt.barh(range(top_n), top_features['importance'])
        plt.yticks(range(top_n), top_features['feature'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Most Important Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    print("Random Forest Predictive Maintenance Model")
    print("=" * 50)
    
    # Initialize model
    rf_model = RandomForestPredictiveModel(
        n_estimators=100,
        max_depth=20,
        random_state=42
    )
    
    # Create dummy data for demonstration
    print("\nCreating sample data...")
    X_train = np.random.randn(1000, 10)
    y_train = np.random.randint(0, 2, 1000)
    X_test = np.random.randn(200, 10)
    y_test = np.random.randint(0, 2, 200)
    
    feature_names = [
        'temperature', 'vibration', 'pressure', 'rpm', 'oil_level',
        'current', 'voltage', 'noise_level', 'humidity', 'runtime_hours'
    ]
    
    # Train model
    rf_model.train(X_train, y_train, feature_names)
    
    # Evaluate model
    metrics = rf_model.evaluate(X_test, y_test)
    
    # Show feature importance
    print("\nTop 5 Most Important Features:")
    print(rf_model.get_feature_importance(5))
    
    print("\nModel ready for deployment!")
