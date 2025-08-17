# NASA_Anomaly_Detection_Using_Vibration_Sensing_Data

An intelligent pipeline for detecting anomalies in NASA's vibration sensing data. This end-to-end system features data loading from MySQL, automated preprocessing (smoothing, scaling, imputation), and advanced outlier detection using Isolation Forest, Z-score, IQR, and LSTM Autoencoders. The best-performing model is saved in a dedicated `model_library`.

## 🚀 Features

- 🔍 Auto-detects and processes sensor anomalies using statistical and deep learning techniques.
- 🧠 Intelligent pipeline selects the best model using correlation benchmarking.
- 💾 Saves models (`.joblib` or `.keras`) into `model_library/` automatically.
- 🌐 HTML frontend, Flask backend, MySQL database integration.

## 🛠️ Tech Stack

- **Frontend:** HTML
- **Backend:** Flask (Python)
- **Database:** MySQL
- **ML/AI:** Scikit-learn, TensorFlow, statsmodels, FLAML
- **Visualization:** Matplotlib, Seaborn
