# 🔥 Industrial Boiler ML Analytics Dashboard

An interactive machine learning dashboard for analyzing **multivariate industrial boiler sensor data**.

The project applies both **unsupervised and supervised machine learning** techniques to identify abnormal operating conditions, discover different operating modes, and forecast future sensor readings.

Built as part of an **AIML course project** using Python, Scikit-learn, and Streamlit.

---

## 📌 Project Overview

Industrial boilers generate large amounts of sensor data containing measurements such as:

* Pressure
* Temperature
* Flow rates
* Oxygen levels
* Fan motor current
* Fan vibration
* Steam flow and temperature

The dataset used in this project contains **unlabeled multivariate time-series data** collected from an industrial boiler SCADA system.

Since the dataset does **not** contain failure labels, maintenance records, efficiency labels, or Remaining Useful Life (RUL) values, the project focuses only on machine learning tasks that can be realistically supported by the available data.

The dashboard performs three main ML tasks:

1. **Anomaly Detection** using Isolation Forest
2. **Operating Mode Discovery** using K-Means Clustering and PCA
3. **Sensor Forecasting** using Random Forest Regression and Linear Regression

---

# 🤖 Machine Learning Tasks

## 1️⃣ Anomaly Detection — Isolation Forest

Industrial sensor data can contain unusual operating conditions caused by abnormal combinations of multiple sensor readings.

This project uses **Isolation Forest**, an unsupervised anomaly detection algorithm, to identify observations that are statistically different from normal operating behavior.

### Input

All available numerical sensor readings.

### Output

Each timestamp is classified as:

* Normal
* Anomalous

The model does **not** predict equipment failure. It identifies statistically unusual sensor patterns.

### Model

```text
Isolation Forest
```

The dashboard displays:

* Anomaly scores over time
* Flagged sensor observations
* Percentage of anomalous observations
* Comparison of sensor behavior during normal and anomalous periods

---

## 2️⃣ Operating Mode Discovery — K-Means Clustering

Industrial boilers can operate under different conditions depending on load, temperature, pressure, and flow requirements.

This project uses **K-Means Clustering** to automatically group similar sensor states into different operating modes.

Before clustering, the sensor data is standardized using:

```text
StandardScaler
```

### Selecting the Number of Clusters

Instead of choosing the number of clusters arbitrarily, multiple values of **K (2–7)** are evaluated.

The following clustering metrics are used:

* **Silhouette Score** — higher values indicate better-separated clusters
* **Davies-Bouldin Score** — lower values indicate better cluster separation

The best value of K is automatically selected using the Silhouette Score.

### Visualization

**Principal Component Analysis (PCA)** is used to reduce the high-dimensional sensor data to two dimensions for visualization.

The dashboard displays:

* Silhouette Score vs K
* Davies-Bouldin Score vs K
* PCA visualization of operating modes
* Distribution of operating modes
* Operating mode changes over time

---

## 3️⃣ Sensor Forecasting — Supervised Machine Learning

The dataset does not contain predefined target labels. Therefore, a supervised learning problem is created directly from the time-series data.

Historical sensor readings are used as input features to predict a **future value of a selected sensor**.

### Example

```text
Sensor readings at:

t-1
t-2
t-3

        ↓

Machine Learning Model

        ↓

Sensor value at:

t + 12
```

### Feature Engineering

Lag features are created from historical readings of all sensors.

For example:

```text
Temperature (t-1)
Temperature (t-2)
Temperature (t-3)

Pressure (t-1)
Pressure (t-2)
Pressure (t-3)

Flow Rate (t-1)
Flow Rate (t-2)
Flow Rate (t-3)
```

These historical readings are used as features to predict the future value of a selected sensor.

### Models Compared

* **Random Forest Regressor**
* **Linear Regression**

### Evaluation Metrics

The models are evaluated using:

* **Mean Absolute Error (MAE)**
* **Root Mean Squared Error (RMSE)**
* **R² Score**

### Preventing Data Leakage

Because this is time-series data, the dataset is split chronologically:

```text
Past Data  → Training Set
Recent Data → Test Set
```

The data is **not shuffled**, preventing the model from accessing future information during training.

The feature scaler is also fitted only on the training data before being applied to the test set.

---

# 📊 Dashboard Features

The Streamlit dashboard provides interactive controls for:

### 🚨 Anomaly Detection

* Expected anomaly fraction
* Isolation Forest analysis
* Anomaly visualization

### 🔁 Operating Mode Discovery

* Automatic cluster selection
* Manual K selection
* Silhouette and Davies-Bouldin evaluation
* PCA visualization

### 📈 Sensor Forecasting

* Sensor selection
* Number of lag features
* Forecast horizon
* Test set size
* Random Forest vs Linear Regression comparison

### 📋 Raw Data Explorer

Users can also:

* Select sensors for visualization
* Explore recent sensor readings
* Visualize raw time-series data

---

# 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Streamlit**
* **Plotly**

---

# 📂 Project Structure

```text
industrial-boiler-ml-dashboard/
│
├── app.py
├── boiler_data.csv
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd industrial-boiler-ml-dashboard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 📈 Dataset

The dataset consists of **multivariate industrial boiler sensor data** sampled over time.

It contains approximately **30 numerical sensor channels**, including measurements related to:

* Furnace pressure
* Temperature
* Flow rates
* Flue gas oxygen
* Steam flow
* Steam temperature
* Induced draft fan current
* Fan vibration

The dataset is **unlabeled**, meaning it does not contain:

* Failure labels
* Maintenance events
* Equipment health scores
* Efficiency labels
* Remaining Useful Life (RUL) values

For this reason, the project does not claim to perform failure prediction or RUL prediction.

---

# ⚠️ Important Note

The anomaly detection model identifies **statistical anomalies**, not confirmed equipment failures.

Similarly, the forecasting model predicts **future sensor values** and does not predict:

* Equipment failure
* Boiler efficiency
* Fan health
* Remaining Useful Life

The objective of this project is to demonstrate how machine learning techniques can be applied responsibly to **unlabeled industrial sensor time-series data**.

---

# 🚀 Future Improvements

Possible future improvements include:

* Adding a naive forecasting baseline for comparison
* Using XGBoost or Gradient Boosting for forecasting
* Implementing LSTM/GRU-based time-series forecasting
* Adding real maintenance and failure labels
* Developing predictive maintenance models
* Performing hyperparameter tuning
* Adding automated model performance tracking

---

## 👤 Author

**Sujal Jaiswal**
230107079

AIML Course Project
