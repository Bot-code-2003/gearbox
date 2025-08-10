# Wind Turbine Gearbox Temperature Prediction System

A deep learning-based early warning system for predicting wind turbine gearbox failures through temperature forecasting. 
The system provides 48-hour ahead predictions with batch processing capabilities through an interactive Streamlit dashboard 
and a FastAPI-based inference API.

## 🚀 Features

- **48-hour temperature forecasting** using LSTM-based deep learning
- **Real-time batch processing** with interactive visualization
- **Multi-threshold warning system** (Warning: 65°C, Failure: 70°C)
- **High accuracy**: 80.86% overall accuracy with 58% recall for failure detection
- **Interactive dashboard** with filtering and export capabilities
- **FastAPI inference endpoint** for programmatic access to predictions

## 📁 Project Structure

```
wind-turbine-prediction/
├── 2_day_gearbox.ipynb          # Main training notebook
├── app.py                       # Streamlit frontend application
├── inference_api.py             # FastAPI inference server
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

### 2. Deployment Phase

#### Option A: Streamlit Dashboard (`app.py`)
- Loads trained model and scalers
- Imports processed test data
- Provides interactive batch prediction interface
- Visualizes results with risk assessment

#### Option B: FastAPI Inference API (`inference_api.py`)
- Exposes a REST API for prediction queries
- Allows specifying:
  - Timestamp to start prediction
  - Lookback steps (default: 384)
  - Forecast steps (default: 192)
  - Critical temperature threshold (default: 65°C)
- Returns predicted values and risk assessment

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.12.1 (recommended for TensorFlow compatibility)
- Git (optional, for cloning)

### Step 1: Environment Setup

```bash
python -m venv venv
# Activate:
# Windows:
venv\Scripts\activate
# Mac/Linux:
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
- `inference_api.py`
- `wt84_with_alarms.csv`
- `test_data_processed.csv`

## 🚀 Running the Application

### Option 1: Training + Dashboard Deployment

```bash
jupyter notebook 2_day_gearbox.ipynb
# Run all cells to train and export model files
streamlit run app.py
```

### Option 2: Dashboard with Pre-trained Model

```bash
streamlit run app.py
```
Access at: `http://localhost:8501`

### Option 3: FastAPI Inference API

```bash
uvicorn inference_api:app --reload --port 8000
```
Access interactive API docs at: `http://127.0.0.1:8000/docs`

#### Example Request (JSON Body)
```json
{
  "timestamp": "2014-11-15 06:30:00",
  "lookback_steps": 384,
  "forecast_steps": 192,
  "critical_temp": 65.0
}
```

#### Example `curl` Command
```bash
curl -X POST "http://127.0.0.1:8000/predict"      -H "Content-Type: application/json"      -d '{"timestamp": "2014-11-15 06:30:00", "critical_temp": 65.0}'
```

## 📊 Dashboard Usage

1. Load Data
2. Configure batch processing
3. Start Processing
4. Monitor results
5. Export CSV

## 🎯 Model Performance

| Metric    | Training | Validation | Test  |
| --------- | -------- | ---------- | ----- |
| MAE (°C)  | 5.385    | 8.153      | 5.911 |
| RMSE (°C) | 8.506    | 10.169     | 8.163 |
| R²        | 0.533    | 0.680      | 0.738 |

Failure detection:
- Precision: 80%
- Recall: 58%
- Accuracy: 80.86%

## 🔧 Technical Details

- **Architecture**: Conv1D + Bidirectional LSTM + Dense layers
- **Lookback**: 96 hrs (384 steps)
- **Forecast**: 48 hrs (192 steps)
- **Resolution**: 15 min
- **Features**: 23 engineered features

## 📈 Future Enhancements

- Weather data integration
- Multi-turbine monitoring
- Automated scheduling
- Real-time streaming
- Mobile UI
