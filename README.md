# Forex Price Prediction ML Model

A complete, end-to-end Machine Learning project that predicts the **next-day closing exchange rate of USD/LKR** using historical Forex data. Two regression models are trained and compared: **Random Forest Regressor** and **Support Vector Regressor (SVR)**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models & Evaluation](#models--evaluation)
- [Results](#results)
- [Technologies Used](#technologies-used)

---

## Project Overview

| Item | Detail |
|---|---|
| **Goal** | Predict next-day USD/LKR closing rate |
| **Problem Type** | Supervised Learning → Regression → Time-Series Forecasting |
| **Target Variable** | `Close.shift(-1)` (next-day close) |
| **Algorithms** | Random Forest Regressor, SVR (RBF kernel) |
| **Evaluation** | MAE, RMSE, R² Score |

---

## Dataset

**Source:** [Kaggle – Currency Exchange Rates](https://www.kaggle.com/datasets/thebasss/currency-exchange-rates)

The dataset contains daily exchange rates for multiple currency pairs. This project filters and uses only the **USD/LKR** pair.

---

## Project Structure

```
forex-price-prediction-ml-model/
├── data/                          # Raw & processed data (downloaded at runtime)
├── models/                        # Saved trained models (.pkl)
├── notebooks/                     # Jupyter notebooks for exploration
├── results/                       # Evaluation metrics & plots
├── src/                           # Source modules
│   ├── __init__.py
│   ├── data_preprocessing.py      # Data loading & cleaning
│   ├── feature_engineering.py     # Feature creation & target variable
│   ├── model_training.py          # Model training & persistence
│   ├── model_evaluation.py        # Metrics calculation
│   └── visualization.py           # Plotting utilities
├── main.py                        # Pipeline entry point
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/forex-price-prediction-ml-model.git
cd forex-price-prediction-ml-model

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

Run the complete pipeline with a single command:

```bash
python main.py
```

This will:
1. Download the dataset (via `kagglehub`)
2. Preprocess and filter USD/LKR data
3. Engineer features (moving averages, rate of change, etc.)
4. Split data chronologically (80 / 20)
5. Train Random Forest and SVR models
6. Evaluate both models (MAE, RMSE, R²)
7. Generate comparison plots and feature-importance charts
8. Save models to `models/` and results to `results/`

---

## Models & Evaluation

| Metric | Random Forest | SVR |
|--------|--------------|-----|
| MAE    | —            | —   |
| RMSE   | —            | —   |
| R²     | —            | —   |

> Results will be populated after the first run and saved to `results/metrics.txt`.

---

## Results

Plots generated after running the pipeline:

- **Actual vs Predicted** comparison chart
- **Feature Importance** bar chart (Random Forest)

All plots are saved in the `results/` directory.

---

## Technologies Used

- Python 3.10+
- pandas, NumPy
- scikit-learn (Random Forest, SVR, StandardScaler)
- Matplotlib, Seaborn
- joblib (model serialization)
- kagglehub (dataset download)

---

## License

This project is open-source and available under the [MIT License](LICENSE).
