import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import time
import logging
import warnings

# Suppress scikit-learn warning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output to console
    ]
)
logger = logging.getLogger(__name__)

# --- Constants ---
MAE = 5
LOOKBACK_STEPS = 384
FORECAST_STEPS = 192
WARNING_THRESHOLD = 65 - MAE
FAILURE_THRESHOLD = 70 - MAE
FORECAST_HOURS = np.arange(0.5, 96.5, 0.5)

# --- Helper Functions ---
def generate_sequences(df, feature_cols, target_col='wtrm_avg_TrmTmp_Gbx', lookback_steps=LOOKBACK_STEPS, forecast_steps=FORECAST_STEPS):
    """Generate sequences from dataframe - same function as training"""
    logger.info(f"Generating sequences with lookback={lookback_steps}, forecast={forecast_steps}")
    X, y, indices = [], [], []
    feature_data = df[feature_cols].values
    target_data = df[target_col].values if target_col in df.columns else None
    dates = df['date_time'].values
    
    for i in range(lookback_steps, len(df) - forecast_steps + 1):
        X.append(feature_data[i - lookback_steps:i])
        if target_data is not None:
            y.append(target_data[i:i + forecast_steps])
        indices.append(dates[i])
    
    return np.array(X), np.array(y) if target_data is not None else None, np.array(indices)

# --- Load Assets ---
@st.cache_resource
def load_assets():
    logger.info("Loading model, scalers, and features")
    try:
        model = load_model('improved_model.h5')
        feature_scaler = joblib.load('feature_scaler.pkl')
        target_scaler = joblib.load('target_scaler.pkl')
        with open('common_features.json') as f:
            common_features = json.load(f)
        logger.info("Assets loaded successfully")
        return model, feature_scaler, target_scaler, common_features
    except Exception as e:
        logger.error(f"Error loading assets: {e}")
        return None, None, None, None

# --- Load and Process Test Data ---
@st.cache_data
def load_and_process_test_data():
    logger.info("Loading and processing test data")
    try:
        df_test = pd.read_csv("test_data_processed.csv", parse_dates=["date_time"])
        X_test, y_test, test_dates = generate_sequences(
            df_test, 
            common_features, 
            target_col='wtrm_avg_TrmTmp_Gbx',
            lookback_steps=LOOKBACK_STEPS,
            forecast_steps=FORECAST_STEPS
        )
        logger.info(f"Test data processed: X_test shape={X_test.shape}")
        return X_test, y_test, test_dates, df_test
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return None, None, None, None

def process_batch(model, feature_scaler, target_scaler, X_batch, common_features):
    """Process a batch of sequences and return max temperatures"""
    logger.info(f"Processing batch of size {X_batch.shape[0]}")
    # Convert to DataFrame to retain feature names
    X_batch_df = pd.DataFrame(
        X_batch.reshape(-1, len(common_features)),
        columns=common_features
    )
    X_batch_scaled = feature_scaler.transform(X_batch_df).reshape(X_batch.shape)
    
    # Predict
    y_pred_scaled = model.predict(X_batch_scaled, verbose=0)
    y_pred_batch = target_scaler.inverse_transform(y_pred_scaled)
    
    # Calculate max, min, avg temperatures for each sequence
    max_temps = np.max(y_pred_batch, axis=1)
    min_temps = np.min(y_pred_batch, axis=1)
    avg_temps = np.mean(y_pred_batch, axis=1)
    
    logger.info(f"Batch processed: max_temp range=[{min(max_temps):.2f}, {max(max_temps):.2f}]")
    return max_temps, min_temps, avg_temps, y_pred_batch

# --- Streamlit UI ---
st.set_page_config(page_title="Batch-wise Gearbox Temp Prediction", layout="wide")
st.title("High-Speed Batch-wise Gearbox Temperature Forecasting")
st.markdown("This app processes data in batches for maximum speed, then visualizes results progressively.")

# Load assets
model, feature_scaler, target_scaler, common_features = load_assets()

if model is None or feature_scaler is None or target_scaler is None or common_features is None:
    st.error("Failed to load required assets. Please check if all files are present.")
    logger.error("Application stopped due to missing assets")
    st.stop()

# Load and process data
with st.spinner("Loading test data..."):
    X_test, y_test, test_dates, df_test = load_and_process_test_data()

if X_test is None:
    st.error("Failed to load test data.")
    logger.error("Application stopped due to failed test data loading")
    st.stop()

# Display data info in sidebar
st.sidebar.markdown("### Dataset Info")
st.sidebar.write(f"Test data shape: {df_test.shape}")
st.sidebar.write(f"Total sequences: {len(X_test)}")
st.sidebar.write(f"Features: {len(common_features)}")
st.sidebar.write(f"Lookback steps: {LOOKBACK_STEPS}")
st.sidebar.write(f"Forecast steps: {FORECAST_STEPS}")

# Batch processing controls
st.sidebar.markdown("### Batch Processing Controls")
batch_size = st.sidebar.selectbox(
    "Batch Size", 
    [10, 25, 50, 100, 200], 
    index=2, 
    help="Number of sequences to process at once"
)

plot_speed = st.sidebar.selectbox(
    "Plotting Speed", 
    [0.01, 0.05, 0.1, 0.2, 0.5], 
    index=1, 
    help="Delay between plotting batches"
)

show_batch_details = st.sidebar.checkbox(
    "Show batch details", 
    value=True, 
    help="Display statistics for each batch"
)

# Calculate number of batches
num_batches = (len(X_test) + batch_size - 1) // batch_size
logger.info(f"Calculated {num_batches} batches for processing")

# Start the batch processing
if st.button("Start Batch Processing", type="primary"):
    
    # Initialize containers
    status_container = st.container()
    plot_container = st.container()
    progress_container = st.container()
    batch_stats_container = st.container()
    
    # Initialize tracking variables
    all_max_temps = []
    all_min_temps = []
    all_avg_temps = []
    all_predictions = []
    processed_dates = []
    
    total_warning_count = 0
    total_failure_count = 0
    total_success_count = 0
    
    # Create progress bar
    with progress_container:
        overall_progress = st.progress(0)
        batch_progress = st.progress(0)
        status_text = st.empty()
    
    # Create plot placeholder
    with plot_container:
        plot_placeholder = st.empty()
    
    # Set up plot limits
    y_min = min(WARNING_THRESHOLD - 10, 45)
    y_max = max(FAILURE_THRESHOLD + 10, 80)
    
    start_time = time.time()
    
    # Process in batches
    for batch_idx in range(num_batches):
        batch_start_time = time.time()
        
        # Calculate batch indices
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(X_test))
        current_batch_size = end_idx - start_idx
        
        # Extract batch
        X_batch = X_test[start_idx:end_idx]
        batch_dates = test_dates[start_idx:end_idx]
        
        # Update status
        with status_container:
            status_text.text(f"Processing batch {batch_idx + 1}/{num_batches} | Sequences {start_idx + 1}-{end_idx}")
        
        # Process batch
        max_temps, min_temps, avg_temps, predictions = process_batch(
            model, feature_scaler, target_scaler, X_batch, common_features
        )
        
        batch_process_time = time.time() - batch_start_time
        logger.info(f"Batch {batch_idx + 1} processed in {batch_process_time:.2f}s")
        
        # Store results
        all_max_temps.extend(max_temps)
        all_min_temps.extend(min_temps)
        all_avg_temps.extend(avg_temps)
        all_predictions.extend(predictions)
        processed_dates.extend(batch_dates)
        
        # Calculate batch statistics
        batch_warnings = np.sum(max_temps > WARNING_THRESHOLD)
        batch_failures = np.sum(max_temps > FAILURE_THRESHOLD)
        batch_success = current_batch_size - batch_warnings
        
        total_warning_count += batch_warnings
        total_failure_count += batch_failures
        total_success_count += batch_success
        
        # Show batch details
        if show_batch_details:
            with batch_stats_container:
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric(f"Batch {batch_idx + 1}", f"{current_batch_size} seq")
                col2.metric("Success", batch_success)
                col3.metric("Warnings", batch_warnings)
                col4.metric("Failures", batch_failures)
                col5.metric("Process Time", f"{batch_process_time:.2f}s")
        
        # Update plot with all processed data so far
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Set up plot
        ax.set_xlim(-2, max(len(X_test) + 2, 50))
        ax.set_ylim(y_min, y_max)
        
        # Background zones
        ax.fill_between(range(-2, len(X_test) + 2), y_min, WARNING_THRESHOLD, 
                       alpha=0.1, color='green', label='Safe Zone')
        ax.fill_between(range(-2, len(X_test) + 2), WARNING_THRESHOLD, FAILURE_THRESHOLD, 
                       alpha=0.1, color='orange', label='Warning Zone')
        ax.fill_between(range(-2, len(X_test) + 2), FAILURE_THRESHOLD, y_max, 
                       alpha=0.1, color='red', label='Failure Zone')
        
        # Threshold lines
        ax.axhline(WARNING_THRESHOLD, color='orange', linestyle='--', linewidth=2, 
                  label=f'Warning ({WARNING_THRESHOLD}°C)', alpha=0.8)
        ax.axhline(FAILURE_THRESHOLD, color='red', linestyle='--', linewidth=2, 
                  label=f'Failure ({FAILURE_THRESHOLD}°C)', alpha=0.8)
        
        # Plot all processed points
        if len(all_max_temps) > 0:
            # Prepare colors for all points
            colors = []
            for temp in all_max_temps:
                if temp > FAILURE_THRESHOLD:
                    colors.append('red')
                elif temp > WARNING_THRESHOLD:
                    colors.append('orange')
                else:
                    colors.append('green')
            
            # Plot line connecting points
            if len(all_max_temps) > 1:
                ax.plot(range(len(all_max_temps)), all_max_temps, color='gray', alpha=0.5, linewidth=1.5, zorder=3)
            
            # Plot all points
            ax.scatter(range(len(all_max_temps)), all_max_temps, c=colors, alpha=0.7, s=25, zorder=5)
            
            # Highlight current batch points
            current_batch_start = len(all_max_temps) - current_batch_size
            current_batch_colors = colors[current_batch_start:]
            ax.scatter(range(current_batch_start, len(all_max_temps)), 
                      all_max_temps[current_batch_start:], 
                      c=current_batch_colors, s=60, alpha=1.0, 
                      edgecolors='black', linewidth=1, zorder=10)
            
            # Show batch boundary
            if batch_idx > 0:
                ax.axvline(current_batch_start, color='blue', linestyle=':', alpha=0.6, linewidth=2)
        
        # Update title
        processed_count = len(all_max_temps)
        ax.set_title(f'Batch {batch_idx + 1}/{num_batches} Complete | Processed: {processed_count}/{len(X_test)} | Success: {total_success_count} Warnings: {total_warning_count} Failures: {total_failure_count}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Sequence Number', fontsize=12)
        ax.set_ylabel('Maximum Temperature (°C)', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Update plot
        plot_placeholder.pyplot(fig)
        plt.close(fig)
        
        # Update progress bars
        overall_progress.progress((batch_idx + 1) / num_batches)
        batch_progress.progress(1.0)  # Batch complete
        
        # Add plotting delay
        if plot_speed > 0:
            time.sleep(plot_speed)
        
        # Reset batch progress for next batch
        if batch_idx < num_batches - 1:
            batch_progress.progress(0.0)
    
    total_time = time.time() - start_time
    logger.info(f"Batch processing completed in {total_time:.2f}s")
    
    # Final status update
    with status_container:
        st.success(f"Batch processing completed! Processed {len(X_test)} sequences in {total_time:.2f} seconds")
        st.info(f"Processing speed: {len(X_test) / total_time:.1f} sequences/second")
    
    # Clear progress
    progress_container.empty()
    
    # Performance metrics
    st.markdown("---")
    st.markdown("### Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Time", f"{total_time:.2f}s")
    col2.metric("Avg Batch Time", f"{total_time / num_batches:.2f}s")
    col3.metric("Processing Speed", f"{len(X_test) / total_time:.1f} seq/s")
    col4.metric("Batches Processed", num_batches)
    
    # Final Statistics
    st.markdown("### Final Results Summary")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Processed", len(X_test))
    col2.metric("Success", total_success_count, delta=f"{(total_success_count/len(X_test)*100):.1f}%")
    col3.metric("Warnings", total_warning_count, delta=f"{(total_warning_count/len(X_test)*100):.1f}%")
    col4.metric("Failures", total_failure_count, delta=f"{(total_failure_count/len(X_test)*100):.1f}%")
    
    # Risk Assessment
    st.markdown("### Risk Assessment")
    warning_pct = (total_warning_count / len(X_test)) * 100
    failure_pct = (total_failure_count / len(X_test)) * 100
    success_pct = (total_success_count / len(X_test)) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Success Rate", f"{success_pct:.1f}%")
        st.progress(min(success_pct / 100, 1.0))
    with col2:
        st.metric("Warning Risk", f"{warning_pct:.1f}%")
        st.progress(min(warning_pct / 100, 1.0))
    with col3:
        st.metric("Failure Risk", f"{failure_pct:.1f}%")
        st.progress(min(failure_pct / 100, 1.0))
    
    # Overall risk status
    if failure_pct > 20:
        st.error("HIGH RISK: More than 20% of forecasts indicate potential failures!")
    elif warning_pct > 50:
        st.warning("MODERATE RISK: More than 50% of forecasts show elevated temperatures!")
    else:
        st.success("LOW RISK: Most forecasts show normal operating conditions!")
    
    # Detailed Results Table
    st.markdown("### Detailed Results")
    
    # Create proper status flags
    warning_flags = [temp > WARNING_THRESHOLD for temp in all_max_temps]
    failure_flags = [temp > FAILURE_THRESHOLD for temp in all_max_temps]
    success_flags = [not w for w in warning_flags]  # Success = not warning
    
    # Create status labels
    status_labels = []
    for i, temp in enumerate(all_max_temps):
        if failure_flags[i]:
            status_labels.append("Failure")
        elif warning_flags[i]:
            status_labels.append("Warning") 
        else:
            status_labels.append("Success")
    
    result_df = pd.DataFrame({
        "Sequence": range(1, len(processed_dates) + 1),
        "Start Time": processed_dates,
        "Max Temp (°C)": [round(temp, 2) for temp in all_max_temps],
        "Min Temp (°C)": [round(temp, 2) for temp in all_min_temps],
        "Avg Temp (°C)": [round(temp, 2) for temp in all_avg_temps],
        "Status": status_labels,
        "Warning Flag": warning_flags,
        "Failure Flag": failure_flags
    })

    st.session_state["result_df"] = result_df

# --- Filter and display results only if result_df is available ---
if "result_df" in st.session_state:
    result_df = st.session_state["result_df"]
    logger.info(f"Displaying results table with {len(result_df)} rows")

    st.markdown("### Detailed Results")
    col1, col2 = st.columns([1, 3])
    with col1:
        filter_option = st.selectbox(
            "Filter results:",
            ["All results", "Success only", "Warnings only", "Failures only"]
        )

    # Apply filter
    if filter_option == "Success only":
        filtered_df = result_df[result_df['Status'] == 'Success'].copy()
    elif filter_option == "Warnings only":
        filtered_df = result_df[result_df['Status'] == 'Warning'].copy()
    elif filter_option == "Failures only":
        filtered_df = result_df[result_df['Status'] == 'Failure'].copy()
    else:
        filtered_df = result_df.copy()

    with col2:
        st.write(f"Showing {len(filtered_df)} of {len(result_df)} sequences")

    # Style the dataframe
    def highlight_rows(df):
        def apply_color(row):
            if row['Status'] == 'Failure':
                return ['background-color: #ffebee; color: #c62828'] * len(row)
            elif row['Status'] == 'Warning':
                return ['background-color: #fff3e0; color: #ef6c00'] * len(row)
            else:
                return ['background-color: #e8f5e8; color: #2e7d32'] * len(row)
        return df.style.apply(apply_color, axis=1)

    styled_df = highlight_rows(filtered_df)
    st.dataframe(styled_df, use_container_width=True, height=400)

    # Download button
    csv_data = filtered_df.to_csv(index=False)
    filename = f"batch_predictions_{filter_option.lower().replace(' ', '_')}.csv"
    st.download_button(
        f"Download {filter_option} ({len(filtered_df)} rows)",
        csv_data,
        filename,
        "text/csv",
        key=f"download_{filter_option}"
    )

else:
    st.info("Click the button above to start high-speed batch processing!")
    st.markdown("""
    ### How Batch Processing Works:
    This app forecasts gearbox temperatures for the next 2 days (192 time steps i.e every 15 min).
                
    **Here's what happens:**
    - **Data Splitting**: Sequences are grouped into batches (e.g., 200 sequences at a time).
    - **Model Prediction**: Each sequence is passed to the model to forecast the next 2 days of temperature.
    - **Max Temp Check**: For each sequence, we take the **maximum predicted temperature**:
    - **Original threshold**: 70°C
    - **Threshold for prediction**: 70 - MAE (°C) to make the predictions more reliable.      
    """)
