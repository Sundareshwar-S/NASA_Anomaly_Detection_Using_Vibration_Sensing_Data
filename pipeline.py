# FILE: pipeline.py
# PURPOSE: This file contains the final, corrected profiler logic in MODULE 3.
# MODULE 1 and MODULE 2 are exactly as you approved them.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
import mysql.connector

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.stattools import adfuller
from scipy.signal import savgol_filter
from scipy.stats import linregress # Import for linear regression
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam

sns.set(style='whitegrid')

#################################################################
## MODULE 1: DATA LOADING UTILITY
#################################################################

def load_data_from_mysql(db_config, num_tables_to_load=1):
    """Connects to MySQL, discovers tables, and loads data from the most recent N tables."""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES;")
        all_tables = sorted([table[0] for table in cursor.fetchall()], reverse=True)
        if not all_tables: raise ValueError("No tables found in the database.")
        
        tables_to_load = all_tables[:num_tables_to_load]
        print(f"--> Loading data from: {tables_to_load}")

        df_list = [pd.read_sql(f"SELECT * FROM `{table}`", conn) for table in reversed(tables_to_load)]
        conn.close()
        
        full_df = pd.concat(df_list, axis=0, ignore_index=True)
        
        if full_df.shape[1] == 8:
            full_df.columns = ['B1_x', 'B1_y', 'B2_x', 'B2_y', 'B3_x', 'B3_y', 'B4_x', 'B4_y']
        elif full_df.shape[1] == 4:
            full_df.columns = ['B1_x', 'B2_x', 'B3_x', 'B4_x']
        else:
            full_df.columns = [f'Sensor_{i+1}' for i in range(full_df.shape[1])]

        print(f"✅ Data loaded successfully. Assigned columns: {full_df.columns.tolist()}")
        
        num_readings = len(full_df)
        full_df.index = pd.to_datetime(pd.date_range(start='2003-10-22 12:06:25', periods=num_readings, freq='ms'))
        
        return full_df
    except Exception as e:
        print(f"Database Error: {e}")
        return None

#################################################################
## MODULE 2: SPECIALIST TOOLBOX (YOUR APPROVED CODE - UNCHANGED)
#################################################################

class SimpleImputation:
    def __init__(self, df):
        self.df = df.copy()
        self.best_method_name = None
        self.methods = {
            'forward_fill': lambda df: df.ffill(),
            'backward_fill': lambda df: df.bfill(),
            'linear_interp': lambda df: df.interpolate(method='linear'),
        }
        print("✅ SimpleImputation initialized.")

    def auto_impute(self):
        if self.df.isnull().sum().sum() == 0:
            print("No missing values to impute.")
            return self.df
        self.best_method_name = 'backward_fill'
        print(f"--> Automated Decision: Selecting '{self.best_method_name}' as the best strategy for initial NaNs.")
        imputed_df = self.methods[self.best_method_name](self.df)
        imputed_df.dropna(inplace=True) 
        return imputed_df

class SmoothingTechniques:
    def __init__(self, df):
        self.df = df.copy()
        self.best_method_name = None
        self.methods = {
            'moving_average': lambda s: s.rolling(window=51, center=True, min_periods=1).mean(),
            'ema': lambda s: s.ewm(span=51, adjust=False).mean(),
            'savgol_filter': lambda s: savgol_filter(s, window_length=51, polyorder=3)
        }
        print("✅ SmoothingTechniques initialized.")

    def auto_smooth(self, column):
        print(f"\nBenchmarking smoothing techniques for column '{column}'...")
        results = {}
        original_signal = self.df[column]
        for name, func in self.methods.items():
            smoothed_signal = func(original_signal)
            noise = original_signal - smoothed_signal
            snr = np.var(smoothed_signal) / np.var(noise) if np.var(noise) > 0 else np.inf
            results[name] = snr
            print(f"  - {name}: SNR = {snr:.2f}")
        self.best_method_name = max(results, key=results.get)
        print(f"--> Automated Decision: Selecting '{self.best_method_name}' as the best smoothing method (Highest SNR).")
        self.df[f"{column}_smoothed"] = self.methods[self.best_method_name](original_signal)
        return self.df

class NormalizationScaling:
    def __init__(self, df):
        self.df = df.copy()
        self.best_method_name = None
        self.scalers = {
            'min_max': MinMaxScaler(),
            'z_score': StandardScaler(),
            'robust': RobustScaler()
        }
        print("✅ NormalizationScaling initialized.")

    def auto_scale(self, column, goal='anomaly_detection'):
        if goal == 'anomaly_detection':
            self.best_method_name = 'min_max'
            print(f"--> Automated Decision: For anomaly detection, selecting '{self.best_method_name}' for consistent range.")
        else:
            self.best_method_name = 'min_max'
        scaler = self.scalers[self.best_method_name]
        self.df[f"{column}_scaled"] = scaler.fit_transform(self.df[[column]])
        return self.df

class IrregularTimeIntervalHandling:
    def __init__(self, df): self.df = df.copy()
    def auto_handle(self):
        is_regular = self._is_regular()
        print(f"Time Interval Regularity Check: {'Regular' if is_regular else 'Irregular'}")
        if is_regular:
            print("--> Automated Decision: Time series is regular. No resampling needed.")
            return self.df
        else:
            print("--> Automated Decision: Time series is irregular. Applying 'Time Resampling'.")
            freq = pd.infer_freq(self.df.index) or 'S' 
            return self.df.resample(freq).mean().interpolate(method='linear')
    def _is_regular(self):
        if not isinstance(self.df.index, pd.DatetimeIndex): return False
        return self.df.index.to_series().diff().dropna().nunique() <= 1

class ModelLibrary:
    def __init__(self, library_path='model_library'):
        self.path = library_path
        if not os.path.exists(self.path): os.makedirs(self.path)
    def save_sklearn_model(self, model, name):
        filepath = os.path.join(self.path, f"{name}.joblib"); joblib.dump(model, filepath)
        print(f"✅ Saved model: {name} to {filepath}")
    def save_keras_model(self, model, name):
        filepath = os.path.join(self.path, f"{name}.keras"); model.save(filepath)
        print(f"✅ Saved model: {name} to {filepath}")

class OutlierDetection:
    def __init__(self, data_series, model_library):
        self.df = pd.DataFrame(data_series).copy()
        self.feature_name = data_series.name
        self.library = model_library
        healthy_split = int(len(self.df) * 0.25)
        self.healthy_data = self.df.iloc[:healthy_split]
        self.X = self.df[[self.feature_name]].values
        self.X_healthy = self.healthy_data[[self.feature_name]].values
        self.best_model_name = None
        self.anomaly_scores = {}

    def run_all_techniques(self):
        print("\n--- Executing 'Outlier Detection' Suite ---")
        self._detect_zscore(); self._detect_iqr(); self._detect_isolation_forest(); self._detect_autoencoder()
        return self.df

    def auto_select_best(self):
        print("\n--- Benchmarking 'Outlier Detection' Models ---")
        correlations = {}
        for name, scores in self.anomaly_scores.items():
            corr = self.df[self.feature_name].corr(pd.Series(scores, index=self.df.index))
            correlations[name] = abs(corr)
            print(f"  - {name}: Correlation Score = {abs(corr):.4f}")
        self.best_model_name = max(correlations, key=correlations.get)
        print(f"--> Automated Decision: Selecting '{self.best_model_name}' as the best model (Highest Correlation).")
        self.df['anomaly_best'] = self.df[f'anomaly_{self.best_model_name}']
        if hasattr(self, f"{self.best_model_name}_threshold"):
             self.best_threshold = getattr(self, f"{self.best_model_name}_threshold")
        return self.df

    def _detect_zscore(self):
        self.zscore_threshold = np.mean(self.X_healthy) + 3 * np.std(self.X_healthy)
        self.df['anomaly_zscore'] = self.df[self.feature_name] > self.zscore_threshold
        self.anomaly_scores['zscore'] = self.X.flatten()

    def _detect_iqr(self):
        Q1, Q3 = np.percentile(self.X_healthy, [25, 75]); IQR = Q3 - Q1
        self.iqr_threshold = Q3 + 1.5 * IQR
        self.df['anomaly_iqr'] = self.df[self.feature_name] > self.iqr_threshold
        self.anomaly_scores['iqr'] = self.X.flatten()

    def _detect_isolation_forest(self):
        model = IsolationForest(n_estimators=100, contamination='auto', random_state=42).fit(self.X_healthy)
        self.library.save_sklearn_model(model, 'isolation_forest_model')
        self.df['anomaly_iso_forest'] = (model.predict(self.X) == -1)
        self.anomaly_scores['iso_forest'] = model.decision_function(self.X)

    def _detect_autoencoder(self, time_steps=64, epochs=3):
        X_seq = np.array([self.X[i:(i + time_steps)] for i in range(len(self.X) - time_steps)])
        X_train_seq = np.array([self.X_healthy[i:(i + time_steps)] for i in range(len(self.X_healthy) - time_steps)])
        inputs = Input(shape=(time_steps, 1)); L1 = LSTM(32, activation='relu', return_sequences=True)(inputs)
        L2 = LSTM(16, activation='relu', return_sequences=False)(L1); L3 = RepeatVector(time_steps)(L2)
        L4 = LSTM(16, activation='relu', return_sequences=True)(L3); L5 = LSTM(32, activation='relu', return_sequences=True)(L4)
        output = TimeDistributed(Dense(1))(L5); model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='mae')
        print("Training LSTM Autoencoder..."); model.fit(X_train_seq, X_train_seq, epochs=epochs, batch_size=32, verbose=1)
        self.library.save_keras_model(model, 'lstm_autoencoder_model')
        train_mae_loss = np.mean(np.abs(model.predict(X_train_seq, verbose=0) - X_train_seq), axis=1)
        self.lstm_threshold = np.percentile(train_mae_loss, 99)
        all_mae_loss = np.mean(np.abs(model.predict(X_seq, verbose=0) - X_seq), axis=1)
        reconstruction_error = np.concatenate([np.zeros(time_steps), all_mae_loss.flatten()])
        self.df['anomaly_lstm'] = reconstruction_error > self.lstm_threshold
        self.anomaly_scores['lstm'] = reconstruction_error

#################################################################
## MODULE 3: INTELLIGENT PIPELINE MANAGER (FINAL CORRECTED VERSION)
#################################################################

class PipelineManager:
    """Analyzes data and automatically selects and runs the appropriate analysis pipeline."""
    def __init__(self, df, target_column):
        self.df = df.copy(); self.target = target_column
        self.profile = {}; self.model_library = ModelLibrary()
        print("✅ PipelineManager initialized.")

    def _run_characteristic_profiler(self, series):
        """
        Final, more robust profiler that uses a targeted heuristic for run-to-failure data.
        """
        print("\n--- Running Final, Robust Characteristic Profiler ---")
        
        # --- DEFINITIVE FIX STARTS HERE ---
        # A more robust method is to check the slope of a linear regression line.
        # This considers the entire series and is less prone to errors from noise.
        
        # Create a simple numeric index for regression
        x = np.arange(len(series))
        y = series.values
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        # A positive slope indicates an upward trend. We check if it's statistically significant.
        has_significant_trend = slope > 0 and p_value < 0.05
        
        print(f"Linear Regression Trend Check: Slope={slope:.6f}, p-value={p_value:.3f}")

        if has_significant_trend:
            self.profile['characteristic'] = "Non-Stationary + Trend"
        else:
            # Fallback to ADF test if the trend isn't clear
            adf_result = adfuller(series.dropna())
            is_statistically_non_stationary = adf_result[1] > 0.05
            if is_statistically_non_stationary:
                self.profile['characteristic'] = "Non-Stationary + Trend"
            else:
                self.profile['characteristic'] = "Stationary"
        # --- DEFINITIVE FIX ENDS HERE ---
            
        print(f"Profiler Conclusion: Data characteristic is '{self.profile['characteristic']}'")

    def execute_pipeline(self, plot_dir='static/plots'):
        """The main orchestration method, adapted for Flask."""
        print("\nStep 1: Feature Engineering..."); feature_name = 'roll_std'
        self.df[feature_name] = self.df[self.target].rolling(window=512).std()
        
        print("\nStep 2: Preprocessing...")
        self.imputer = SimpleImputation(self.df); self.df = self.imputer.auto_impute()
        self.interval_handler = IrregularTimeIntervalHandling(self.df); self.df = self.interval_handler.auto_handle()
        self.smoother = SmoothingTechniques(self.df); self.df = self.smoother.auto_smooth(column=feature_name)
        self.scaler = NormalizationScaling(self.df); self.df = self.scaler.auto_scale(column=f"{feature_name}_smoothed")
        final_feature = f"{feature_name}_smoothed_scaled"
        
        self._run_characteristic_profiler(self.df[final_feature])
        
        print("\n--- Automated Pipeline Decision ---")
        if self.profile['characteristic'] == "Non-Stationary + Trend":
            print("✅ Decision: Characteristic matches 'Non-Stationary + Trend'.")
            print("--> Action: Executing 'OutlierDetection' suite.")
            subsample_rate = 20
            df_sampled = self.df.iloc[::subsample_rate, :].copy()
            health_indicator = df_sampled[final_feature]
            health_indicator.name = 'Processed Health Indicator'
            
            self.executor = OutlierDetection(health_indicator, self.model_library)
            self.results_df = self.executor.run_all_techniques()
            self.results_df = self.executor.auto_select_best()
            
            plot_paths = self.visualize_results(plot_dir)
            report = self._generate_final_report()
            return report, plot_paths
        else:
            report = "Decision: No pipeline matched for the detected characteristic. Halting."
            return report, []

    def visualize_results(self, plot_dir):
        """Saves plots to files and returns their paths."""
        print("\n--- Generating and Saving Plots ---")
        plot_paths = []
        anomaly_cols = [col for col in self.results_df.columns if 'anomaly_' in col]
        
        plot_titles = {'anomaly_zscore': 'Technique: Z-Score', 'anomaly_iqr': 'Technique: IQR',
                       'anomaly_iso_forest': 'Technique: Isolation Forest', 'anomaly_lstm': 'Technique: LSTM Autoencoder',
                       'anomaly_best': f'**Best Model: {self.executor.best_model_name.replace("_", " ").title()}**'}
        
        for i, (col, title) in enumerate(plot_titles.items()):
            if col in self.results_df.columns:
                fig, ax = plt.subplots(figsize=(15, 5))
                if col == 'anomaly_best':
                    threshold = getattr(self.executor, 'best_threshold', None)
                else:
                    threshold = getattr(self.executor, col.replace('anomaly_', '') + '_threshold', None)
                self._plot_anomalies(ax, self.results_df, self.executor.feature_name, col, title, threshold)
                path = os.path.join(plot_dir, f"plot_{self.target}_{i}_{col}.png")
                plt.savefig(path); plt.close(fig)
                plot_paths.append(path)
                print(f"Saved plot: {path}")
        return plot_paths

    def _plot_anomalies(self, ax, df, feature_col, anomaly_col, title, threshold=None):
        ax.plot(df.index, df[feature_col], label='Health Indicator', zorder=1)
        anomalies = df[df[anomaly_col]]
        ax.scatter(anomalies.index, anomalies[feature_col], color='red', label='Anomaly', zorder=2, s=50)
        if threshold: ax.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})', zorder=3)
        ax.set_title(title, fontsize=16); ax.legend()

    def _generate_final_report(self):
        """Generates a human-readable summary of the pipeline's execution and findings."""
        report_lines = [
            "=========================================================",
            f"          FINAL REPORT FOR: {self.target}         ",
            "=========================================================",
            "\n1. Preprocessing Summary:",
            f"   - Missing Values Handled Using: '{self.imputer.best_method_name}'",
            f"   - Smoothing Technique Chosen:   '{self.smoother.best_method_name}'",
            f"   - Scaling Method Chosen:        '{self.scaler.best_method_name}'",
            "\n2. Data Characteristic Analysis:",
            f"   - Final Conclusion: Data is '{self.profile['characteristic']}'",
            "   - Action Taken:     Executed the 'Outlier Detection' model suite.",
            "\n3. Anomaly Detection Results:",
            f"   - Best Performing Model: '{self.executor.best_model_name.replace('_', ' ').title()}'",
        ]
        anomalies_found = self.results_df['anomaly_best'].sum() > 0
        report_lines.append(f"   - Anomalies Detected:      {'YES' if anomalies_found else 'NO'}")
        report_lines.append("\n4. Final Recommendation:")
        if anomalies_found:
            report_lines.append("   - Verdict: 🔴 MAINTENANCE RECOMMENDED. Anomalous patterns indicating potential failure were detected.")
        else:
            report_lines.append("   - Verdict: 🟢 SYSTEM HEALTHY. No significant anomalies were detected.")
        report_lines.append("\n5. Model Artifacts:")
        report_lines.append("   - The best performing model and its components have been saved to the 'model_library' directory.")
        report_lines.append("=========================================================\n")
        return "\n".join(report_lines)
