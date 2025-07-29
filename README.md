# Wind Turbine Gearbox Temperature Prediction System

A deep learning-based early warning system for predicting wind turbine gearbox failures through temperature forecasting. The system provides 48-hour ahead predictions with batch processing capabilities through an interactive Streamlit dashboard.

## 🚀 Features

- **48-hour temperature forecasting** using LSTM-based deep learning
- **Real-time batch processing** with interactive visualization
- **Multi-threshold warning system** (Warning: 65°C, Failure: 70°C)
- **High accuracy**: 80.86% overall accuracy with 58% recall for failure detection
- **Interactive dashboard** with filtering and export capabilities

## 📁 Project Structure

```
wind-turbine-prediction/
├── 2_day_gearbox.ipynb          # Main training notebook
├── app.py                       # Streamlit frontend application
├── wt84_with_alarms.csv         # Original training dataset
├── test_data_processed.csv      # Processed test data for deployment
├── improved_model.h5            # Trained LSTM model (generated)
├── feature_scaler.pkl           # Feature scaler (generated)
├── target_scaler.pkl            # Target scaler (generated)
├── common_features.json         # Feature list (generated)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔄 Model Training to Deployment Workflow

### 1. Training Phase (`2_day_gearbox.ipynb`)

- Loads raw wind turbine data from `wt84_with_alarms.csv`
- Performs data preprocessing and 15-minute aggregation
- Engineers features (alarm features, lag features, cyclic time features)
- Trains Conv1D + LSTM hybrid model
- **Exports**:
  - `improved_model.h5` - Trained model
  - `feature_scaler.pkl` - Feature normalization scaler
  - `target_scaler.pkl` - Target temperature scaler
  - `common_features.json` - List of features used
  - `test_data_processed.csv` - Processed test data

### 2. Deployment Phase (`app.py`)

- Loads trained model and scalers
- Imports processed test data
- Provides interactive batch prediction interface
- Visualizes results with risk assessment

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.12.1 (recommended for TensorFlow compatibility)
- Git (optional, for cloning)

### Step 1: Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Files

Ensure these files are in your project directory:

- `2_day_gearbox.ipynb`
- `app.py`
- `wt84_with_alarms.csv`
- `test_data_processed.csv`

## 🚀 Running the Application

### Option 1: Training + Deployment (Full Pipeline)

1. **Train the model first:**

```bash
# Open Jupyter notebook
jupyter notebook 2_day_gearbox.ipynb

# Run all cells to:
# - Process data
# - Train model
# - Generate model files (improved_model.h5, scalers, etc.)
```

2. **Launch the dashboard:**

```bash
streamlit run app.py
```

### Option 2: Deployment Only (Pre-trained Model)

If you already have the trained model files:

```bash
streamlit run app.py
```

The dashboard will be available at: `http://localhost:8501`

## 📊 Using the Dashboard

1. **Load Data**: The app automatically loads processed test data
2. **Configure Batch Processing**:
   - Select batch size (10–200 sequences)
   - Choose plotting speed (0.01–0.5s delay)
   - Toggle batch details display
3. **Start Processing**: Click "🚀 Start Batch Processing"
4. **Monitor Results**: View real-time predictions and risk assessment
5. **Export Results**: Download filtered results as CSV

## 🎯 Model Performance

| Metric    | Training | Validation | Test  |
| --------- | -------- | ---------- | ----- |
| MAE (°C)  | 5.385    | 8.153      | 5.911 |
| RMSE (°C) | 8.506    | 10.169     | 8.163 |
| R²        | 0.533    | 0.680      | 0.738 |

### Failure Detection Performance:

- **Precision**: 80% (4 out of 5 predictions are correct)
- **Recall**: 58% (detects 58% of actual failures)
- **Accuracy**: 80.86% (overall correctness)

## 🔧 Technical Details

### Model Architecture:

- **Conv1D**: Local pattern extraction
- **Bidirectional LSTM**: Long-term dependency capture
- **Dense layers**: Final prediction mapping

### Key Features:

- **Lookback window**: 96 hours (384 time steps)
- **Forecast horizon**: 48 hours (192 time steps)
- **Temporal resolution**: 15 minutes
- **Input features**: 23 engineered features including temperature sensors, operational parameters, and alarm indicators

### Thresholds:

- **Warning threshold**: 65°C (Threshold - MAE)
- **Failure threshold**: 70°C (85th percentile)
- **Safe prediction limit**: Accounts for model uncertainty

## 🐛 Troubleshooting

### Common Issues

1. **"Failed to load assets" error**:

   - Run the training notebook first to generate model files
   - Ensure all files are in the same directory

2. **TensorFlow installation issues**:

   - Use Python 3.12.1 for best compatibility
   - Try: `pip install tensorflow==2.10.0`

3. **Memory issues during batch processing**:

   - Reduce batch size to 10–25
   - Close other applications

4. **Slow performance**:
   - Increase batch size to 100–200
   - Reduce plotting speed to 0.01s

## 📈 Future Enhancements

- Weather data integration
- Multi-turbine fleet monitoring
- Automated maintenance scheduling
- Real-time streaming predictions
- Mobile-responsive interface
