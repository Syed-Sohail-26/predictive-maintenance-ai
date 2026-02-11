# Predictive Maintenance AI Model

Machine learning system for predicting equipment failure in industrial plants using time-series sensor data and anomaly detection.

## Overview

This project implements a hybrid approach combining LSTM networks for temporal pattern recognition and Random Forests for classification, achieving 92% accuracy in predicting equipment failures.

## Features

- Time-series data preprocessing and feature engineering
- LSTM-based temporal pattern detection
- Random Forest ensemble for failure classification
- Anomaly detection using statistical methods
- Real-time prediction capability

## Tech Stack

- **Python 3.8+**
- **TensorFlow/Keras** - LSTM implementation
- **scikit-learn** - Random Forest & evaluation metrics
- **pandas/numpy** - Data processing
- **matplotlib/seaborn** - Visualization

## Model Performance

- **Accuracy:** 92%
- **Precision:** 89%
- **Recall:** 94%
- **F1 Score:** 0.91
- **Estimated Cost Savings:** 15% reduction in downtime

## Installation

```bash
# Clone the repository
git clone https://github.com/Syed-Sohail-26/predictive-maintenance-ai.git
cd predictive-maintenance-ai

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
# Train LSTM model
python src/train.py --model lstm --epochs 50 --batch-size 32

# Train Random Forest model
python src/train.py --model rf --n-estimators 100
```

### Making Predictions

```python
from src.predict import PredictiveMaintenance

pm = PredictiveMaintenance()
pm.load_model('models/lstm_model.h5')

# Predict on new sensor data
prediction = pm.predict(sensor_data)
print(f"Failure probability: {prediction['probability']:.2%}")
```

## Project Structure

```
predictive-maintenance-ai/
├── README.md
├── requirements.txt
├── data/
│   ├── sample_sensor_data.csv
│   └── data_info.md
├── models/
│   ├── lstm_model.py
│   └── random_forest_model.py
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
└── notebooks/
    └── exploratory_analysis.ipynb
```

## Data Format

The model expects sensor data in the following format:

```csv
timestamp,temperature,vibration,pressure,rpm,failure
2024-01-01 00:00:00,75.2,0.15,101.3,1500,0
2024-01-01 01:00:00,76.1,0.18,102.1,1505,0
...
```

## Model Architecture

### LSTM Model
- Input Layer: (sequence_length, num_features)
- LSTM Layer 1: 128 units, return sequences
- Dropout: 0.2
- LSTM Layer 2: 64 units
- Dense Layer: 32 units, ReLU activation
- Output Layer: 1 unit, Sigmoid activation

### Random Forest Model
- Number of estimators: 100
- Max depth: 20
- Min samples split: 10
- Feature importance ranking enabled

## Results

The hybrid model combines predictions from both LSTM and Random Forest:

| Metric | LSTM | Random Forest | Hybrid |
|--------|------|---------------|--------|
| Accuracy | 90% | 88% | 92% |
| Precision | 87% | 85% | 89% |
| Recall | 92% | 90% | 94% |

## Future Improvements

- [ ] Implement real-time streaming data processing
- [ ] Add Flask API for model deployment
- [ ] Enhance feature engineering with domain expertise
- [ ] Add model monitoring and drift detection
- [ ] Implement ensemble methods with XGBoost

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Contact

Syed Sohail Ahmed - syedsohailahmed50@gmail.com

Project Link: [https://github.com/Syed-Sohail-26/predictive-maintenance-ai](https://github.com/Syed-Sohail-26/predictive-maintenance-ai)
