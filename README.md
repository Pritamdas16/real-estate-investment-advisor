# real-estate-investment-advisor
# 🏠 Real Estate Investment Advisor

> AI-powered property analysis — Classification + 5-Year Price Forecasting

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green)

---

## 📌 Problem Statement

Develop a machine learning application to assist real estate investors by:
1. **Classifying** whether a property is a *"Good Investment"*
2. **Predicting** the estimated property price after **5 years**

---

## 🎯 Features

- ✅ Investment classification (Good / Not Good)
- 📈 5-year future price prediction
- 📊 20 EDA charts (price trends, location analysis, correlations)
- 🔬 MLflow experiment tracking & model registry
- 🖥️ Interactive Streamlit dashboard

---

## 🗂️ Project Structure

```
real_estate/
├── india_housing_prices.csv     ← raw dataset
├── step1_preprocessing.py       ← data cleaning & feature engineering
├── step2_eda.py                 ← 20 exploratory charts
├── step3_model.py               ← model training & evaluation
├── step4_mlflow.py              ← experiment tracking
├── step5_app.py                 ← Streamlit web app
├── models/
│   ├── classifier.pkl           ← XGBoost classifier
│   ├── regressor.pkl            ← best regression model
│   ├── feature_names.pkl
│   └── imputer.pkl
├── eda_charts/                  ← 20 EDA PNG charts
├── model_charts/                ← evaluation charts
└── mlruns/                      ← MLflow logs
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas / NumPy | Data processing |
| Scikit-learn | ML models |
| XGBoost | Best classifier |
| MLflow | Experiment tracking |
| Streamlit | Web app deployment |
| Matplotlib / Seaborn | Visualizations |

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost mlflow streamlit matplotlib seaborn joblib
```

### 2. Run each step in order
```bash
python step1_preprocessing.py   # Data preprocessing
python step2_eda.py             # EDA charts
python step3_model.py           # Model training
python step4_mlflow.py          # MLflow tracking
streamlit run step5_app.py      # Launch web app
```

### 3. View MLflow UI
```bash
mlflow ui
# Open: http://localhost:5000
```

### 4. View Streamlit App
```
http://localhost:8501
```

---

## 📊 Model Performance

### Classification (Good Investment?)
| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | 85.6% | 0.87 | 0.89 |
| Random Forest | 94.4% | 0.95 | 0.99 |
| **XGBoost ★** | **98.9%** | **0.99** | **1.00** |

### Regression (Future Price 5Y)
| Model | RMSE | R² (CV) |
|-------|------|---------|
| Linear Regression | Best | ~0.99 |
| Random Forest | Good | ~0.98 |
| XGBoost | Good | ~0.99 |

---

## 🔍 Dataset Features

| Feature | Description |
|---------|-------------|
| State / City / Locality | Location info |
| Property_Type | Apartment, Villa, House, Studio |
| BHK | Number of bedrooms |
| Size_in_SqFt | Area in square feet |
| Price_in_Lakhs | Current price |
| Age_of_Property | Property age |
| Nearby_Schools/Hospitals | Amenity count |
| Public_Transport_Accessibility | Low/Medium/High |
| Good_Investment | Target (0/1) |
| Future_Price_5Y | Target (regression) |

---

## 📁 Target Variables

**Classification:** `Good_Investment`
- 1 = Good Investment
- 0 = Not a Good Investment

**Regression:** `Future_Price_5Y`
- Formula: `Price × (1 + appreciation_rate) ^ 5`
- Rate: High city → 10%, Mid → 8%, Low → 6%

---

## 👤 Author

**Pritam Das**
BCA 5th Semester — Raiganj University
GitHub: [Pritamdas16](https://github.com/Pritamdas16)

---

## 📄 License

This project is for educational purposes.
