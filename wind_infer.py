# wind_infer.py

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler, LabelEncoder
import joblib, json, warnings
warnings.filterwarnings("ignore")

class WindTurbineInference:
    """
    EXACT mirror of training, but anchored at a user-selected source point:
      - history is trimmed up to source point
      - builds a 3-day (288 steps @15min) lookback ending at source
      - forecasts next 2 days (192 steps @15min)
      - attaches actuals over the forecast window
    """

    def __init__(self,
                 model_path='improved_model.h5',
                 feature_scaler_path='feature_scaler.pkl',
                 target_scaler_path='target_scaler.pkl',
                 features_path='common_features.json',
                 label_encoder_path='label_encoder_alarm_system.pkl',
                 lookback_steps=288,        # 3 days * 24 * 4 (15-min)
                 forecast_steps=192,        # 2 days * 24 * 4 (15-min)
                 critical_temp=70.0):
        print("🔄 Loading model & scalers...")
        self.model = tf.keras.models.load_model(model_path)
        self.feature_scaler: RobustScaler = joblib.load(feature_scaler_path)
        self.target_scaler: RobustScaler = joblib.load(target_scaler_path)
        with open(features_path, 'r') as f:
            self.common_features = json.load(f)
        try:
            self.label_encoder: LabelEncoder = joblib.load(label_encoder_path)
        except Exception:
            self.label_encoder = None
            print("⚠️ Label encoder not found; will fit/approximate if needed.")

        # Guard: feature order must match scaler, if available
        scaler_feats = getattr(self.feature_scaler, "feature_names_in_", None)
        if scaler_feats is not None and list(scaler_feats) != list(self.common_features):
            raise ValueError("Feature order mismatch between feature_scaler and common_features.json.")

        self.lookback_steps = lookback_steps
        self.forecast_steps = forecast_steps
        self.critical_temp = critical_temp
        print("✅ Inference pipeline ready")

    # ------------------- PUBLIC API ------------------- #
    def predict_from_point(self, latest_point, history, actual_df=None):
        """
        Build a forecast that STARTS RIGHT AFTER the chosen source point.

        latest_point: dict or 1-row DataFrame with 'date_time' + raw sensors
        history: DataFrame or CSV path (5-min-ish raw)
        actual_df: optional DataFrame to attach actual temperatures
        """
        # --- 0) Load raw history --- #
        if isinstance(history, str):
            raw = pd.read_csv(history, parse_dates=['date_time'])
        else:
            raw = history.copy()
        raw['date_time'] = pd.to_datetime(raw['date_time'])

        # latest_point as DataFrame and get source timestamp
        latest_df = pd.DataFrame([latest_point]) if isinstance(latest_point, dict) else latest_point.copy()
        latest_df['date_time'] = pd.to_datetime(latest_df['date_time'])
        if len(latest_df) != 1:
            raise ValueError("latest_point must be a single row.")
        source_ts = latest_df['date_time'].iloc[0]

        # --- 1) Trim history up to source point + ensure source row present --- #
        df = raw[raw['date_time'] <= source_ts]
        df = pd.concat([df, latest_df], ignore_index=True).drop_duplicates(subset='date_time', keep='last')
        df = df.sort_values('date_time')
        print(f"🔄 Initial trimmed shape: {df.shape}")

        # --- 2) Reindex to full 5-min grid (as in training) --- #
        full5 = pd.date_range(df['date_time'].min(), df['date_time'].max(), freq='5min')
        df = df.set_index('date_time').reindex(full5).reset_index().rename(columns={'index':'date_time'})
        df = df.fillna(method='ffill').fillna(method='bfill')
        print(f"✅ After 5-min reindex: {df.shape}")

        # --- 3) Encode alarm_system BEFORE 15-min resample (as in training) --- #
        if 'alarm_system' in df.columns:
            df['alarm_system'] = df['alarm_system'].astype(str)
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder().fit(df['alarm_system'])
            else:
                # map unknowns to first known class
                known = set(self.label_encoder.classes_)
                if not set(df['alarm_system']).issubset(known):
                    df['alarm_system'] = df['alarm_system'].apply(
                        lambda x: x if x in known else list(known)[0]
                    )
            df['alarm_system'] = self.label_encoder.transform(df['alarm_system'])

        # --- 4) Drop rows with missing alarm_desc if your training did so --- #
        if 'alarm_desc' in df.columns:
            df = df.dropna(subset=['alarm_desc'])

        # --- 5) Resample to 15-min (numerics mean, alarm max) --- #
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df_num = df[['date_time'] + num_cols].set_index('date_time').resample('15min').mean()
        if 'alarm_system' in df.columns:
            alarm_15 = df[['date_time','alarm_system']].set_index('date_time').resample('15min').max()
            df_num['alarm_system'] = alarm_15['alarm_system']
        df_15 = df_num.reset_index()

        # --- 6) Feature engineering (same as training) --- #
        df_15 = self._alarm_features(df_15)
        df_15 = self._enhanced_features(df_15)

        # --- 7) Align to source grid time and ensure feature columns --- #
        source_15 = pd.Timestamp(source_ts).floor('15min')
        df_15 = df_15[df_15['date_time'] <= source_15].reset_index(drop=True)

        for f in self.common_features:
            if f not in df_15.columns:
                df_15[f] = 0.0
        # keep only needed cols in correct order
        df_clean = df_15[self.common_features + ['wtrm_avg_TrmTmp_Gbx', 'date_time']].dropna().reset_index(drop=True)
        print(f"✅ Final preprocessed shape (<= source): {df_clean.shape}")

        # --- 8) Build last 3-day lookback ending at source_15 --- #
        X, anchor_time = self._sequence_from_tail(df_clean)
        # If padding occurred, anchor_time is still the last row (source_15).

        # --- 9) Scale features (train scaler, order safe) --- #
        X_scaled = self._scale_features(X).astype(np.float32)

        # --- 10) Predict next 2 days --- #
        y_scaled = self.model.predict(X_scaled, verbose=0)
        if y_scaled.ndim == 3:
            y_scaled = y_scaled.reshape(y_scaled.shape[0], -1)
        # inverse transform expects shape (1, forecast_steps)
        y = self.target_scaler.inverse_transform(y_scaled)[0]

        # --- 11) Forecast timestamps start right after source_15 --- #
        forecast_times = pd.date_range(
            start=source_15 + pd.Timedelta(minutes=15),
            periods=len(y),
            freq='15min'
        )
        preds_df = pd.DataFrame({'timestamp': forecast_times, 'predicted_temperature': y})

        # --- 12) Threshold & summary stats --- #
        exceed_mask = y > self.critical_temp
        exceeded = bool(exceed_mask.any())
        max_idx = int(np.argmax(y))
        max_temp = float(y[max_idx])
        max_time = forecast_times[max_idx]
        first_exceed_time = forecast_times[np.argmax(exceed_mask)] if exceeded else None
        total_exceed_count = int(exceed_mask.sum())

        # --- 13) Attach actuals within forecast window --- #
        preds_with_actuals, overlap_metrics = self._attach_actuals(preds_df, actual_df)

        # --- 14) Package results --- #
        results = {
            'predictions': preds_with_actuals,
            'exceeded': exceeded,
            'first_exceed_time': first_exceed_time,
            'max_temperature': max_temp,
            'max_temperature_time': max_time,
            'critical_temperature_threshold': self.critical_temp,
            'prediction_start': source_15,
            'prediction_end': forecast_times[-1],
            'total_exceed_count': total_exceed_count,
            'overlap_metrics': overlap_metrics
        }

        # --- 15) Pretty log --- #
        self._log_summary(results, source_ts)

        # sanity check
        assert len(y) == self.forecast_steps, "Forecast length mismatch vs forecast_steps."
        return results

    # ------------------- HELPERS (mirror training) ------------------- #
    def _alarm_features(self, df, alarm_col='alarm_system', time_col='date_time'):
        df = df.copy()
        if alarm_col in df.columns:
            alarms = df[df[alarm_col] == 1][time_col].reset_index(drop=True)

            def hours_since(ts):
                past = alarms[alarms < ts]
                return (ts - past.iloc[-1]).total_seconds()/3600 if not past.empty else np.nan

            df['hours_since_last_alarm'] = df[time_col].apply(hours_since).fillna(48)
            df['recent_alarm_flag'] = (df['hours_since_last_alarm'] < 6).astype(int)
            df['alarm_frequency_24h'] = df[alarm_col].rolling(window=96).sum().fillna(0)  # 24h at 15-min
            df['alarm_system_lag_0.5h'] = df[alarm_col].shift(1)
            df['alarm_system_lag_2h']   = df[alarm_col].shift(4)
        else:
            df['hours_since_last_alarm'] = 48.0
            df['recent_alarm_flag'] = 0
            df['alarm_frequency_24h'] = 0.0
            df['alarm_system_lag_0.5h'] = 0
            df['alarm_system_lag_2h'] = 0
        return df

    def _enhanced_features(self, df, target_col='wtrm_avg_TrmTmp_Gbx', time_col='date_time'):
        df = df.copy()

        # Critical temps with lags/deltas
        critical = ['wtrm_avg_TrmTmp_GbxBrg452','wtrm_avg_TrmTmp_GbxBrg151','wtrm_avg_TrmTmp_Gbx']
        lag_steps = [1,2,4,8,16]  # 0.5h,1h,2h,4h,8h
        for col in critical:
            if col in df.columns:
                for lag in lag_steps:
                    df[f'{col}_lag_{lag*0.5}h'] = df[col].shift(lag)
                df[f'{col}_delta_1h'] = df[col] - df[col].shift(2)
                df[f'{col}_delta_4h'] = df[col] - df[col].shift(8)

        if target_col in df.columns:
            df[f'{target_col}_rolling_mean_3.0h'] = df[target_col].rolling(window=6).mean()

        # Ops features
        for col in ['wgen_avg_Spd','wgdc_avg_TriGri_PwrAt','wtrm_avg_Gbx_OilPres']:
            if col in df.columns:
                df[f'{col}_delta_1h'] = df[col] - df[col].shift(2)
                df[f'{col}_rolling_mean_6h'] = df[col].rolling(window=12).mean()

        # Time features
        df['hour'] = df[time_col].dt.hour + df[time_col].dt.minute/60
        df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
        df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
        df['day_of_week'] = df[time_col].dt.dayofweek

        df['week_of_year'] = df[time_col].dt.isocalendar().week.astype(int)
        df['week_sin'] = np.sin(2*np.pi*df['week_of_year']/52)
        df['week_cos'] = np.cos(2*np.pi*df['week_of_year']/52)

        df['month'] = df[time_col].dt.month
        df['month_sin'] = np.sin(2*np.pi*df['month']/12)
        df['month_cos'] = np.cos(2*np.pi*df['month']/12)

        # Clean
        df = df.fillna(method='ffill').fillna(method='bfill')
        df = df.dropna().reset_index(drop=True)
        return df

    def _sequence_from_tail(self, df):
        """Take last lookback_steps (pad if needed) ending at the last row (source_15)."""
        if len(df) < self.lookback_steps:
            pad_needed = self.lookback_steps - len(df)
            last_row = df.iloc[-1:].copy()
            end_time = df['date_time'].iloc[-1]
            pad_times = pd.date_range(end=end_time - pd.Timedelta(minutes=15),
                                      periods=pad_needed, freq='-15min')[::-1]
            pad_df = pd.concat([last_row]*pad_needed, ignore_index=True)
            pad_df['date_time'] = pad_times
            df = pd.concat([pad_df, df], ignore_index=True)

        feat = df[self.common_features].values[-self.lookback_steps:]
        anchor_time = df['date_time'].iloc[-1]  # == source_15
        X = feat.reshape(1, self.lookback_steps, len(self.common_features))
        return X, anchor_time

    def _scale_features(self, X):
        """Transform with training feature_scaler (order already validated)."""
        X2d = X.reshape(-1, X.shape[-1])
        # if scaler has feature_names_in_, ensure same order:
        scaler_feats = getattr(self.feature_scaler, "feature_names_in_", None)
        if scaler_feats is not None and list(scaler_feats) != list(self.common_features):
            raise ValueError("Feature order drift before scaling.")
        Xs = self.feature_scaler.transform(X2d).reshape(X.shape)
        return Xs

    def _attach_actuals(self, preds_df, actual_df, target_col='wtrm_avg_TrmTmp_Gbx'):
        """Attach actuals over the forecast window; robust to small timing drift."""
        if actual_df is None or len(actual_df) == 0:
            return preds_df, {}

        df = actual_df.copy()
        if 'date_time' not in df.columns or target_col not in df.columns:
            return preds_df, {}

        df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df = df.dropna(subset=['date_time'])

        # align to 15-min grid and average duplicates
        df['date_time'] = df['date_time'].dt.floor('15min')
        actuals_15 = (df[['date_time', target_col]]
                        .groupby('date_time', as_index=False)
                        .mean()
                        .rename(columns={'date_time':'timestamp', target_col:'actual_temperature'}))

        # limit to forecast window
        start_ts = preds_df['timestamp'].min()
        end_ts   = preds_df['timestamp'].max()
        window_actuals = actuals_15[(actuals_15['timestamp'] >= start_ts) &
                                    (actuals_15['timestamp'] <= end_ts)].copy()

        preds = preds_df.sort_values('timestamp').copy()
        merged = preds.merge(window_actuals, on='timestamp', how='left')
        exact_cover = merged['actual_temperature'].notna().mean()

        # fallback nearest-with-tolerance if coverage poor
        if exact_cover < 0.8:
            merged = pd.merge_asof(
                preds.sort_values('timestamp'),
                window_actuals.sort_values('timestamp'),
                on='timestamp',
                direction='nearest',
                tolerance=pd.Timedelta(minutes=7, seconds=30)
            )

        mask = merged['actual_temperature'].notna()
        metrics = {}
        if mask.any():
            y_true = merged.loc[mask, 'actual_temperature'].to_numpy()
            y_pred = merged.loc[mask, 'predicted_temperature'].to_numpy()
            mae  = float(np.mean(np.abs(y_true - y_pred)))
            rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
            bias = float(np.mean(y_pred - y_true))
            metrics = {'overlap_points': int(mask.sum()), 'mae': mae, 'rmse': rmse, 'bias': bias}
            merged.loc[mask, 'abs_error'] = np.abs(y_true - y_pred)
        else:
            merged['abs_error'] = np.nan

        return merged, metrics

    def _log_summary(self, results, source_ts):
        print("\n📊 Inference Summary")
        print(f"📍 Source point timestamp: {pd.Timestamp(source_ts)}")
        print(f"🔮 Forecast window: {results['prediction_start']} → {results['prediction_end']}")
        print(f"🌡️ Peak temperature: {results['max_temperature']:.2f}°C at {results['max_temperature_time']}")
        if results['exceeded']:
            # print(f"⚠️ Threshold {results['critical_temperature_threshold']}°C exceeded {results['total_exceed_count']} times")
            print(f"   First exceedance: {results['first_exceed_time']}")
        if results['overlap_metrics']:
            m = results['overlap_metrics']
            print(f"🔄 Overlap: {m['overlap_points']} points | MAE={m['mae']:.3f}°C, RMSE={m['rmse']:.3f}°C, Bias={m['bias']:.3f}°C")

