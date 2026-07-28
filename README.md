# 🎓 Student Performance Prediction ML Project

An end-to-end Machine Learning solution designed to predict whether a student will achieve **High Performance** (Final Score >= 70%) or **Low Performance** (Final Score < 70%), as well as forecast their exact continuous exam score based on academic, personal, and socioeconomic factors.

---

## 📁 Project Structure

```
Student-Performance-Prediction/
│
├── dataset/
│   └── student_performance.csv    # Synthetic student dataset (1,000 records)
│
├── notebook/
│   └── student_performance_eda.ipynb  # Interactive Jupyter notebook for EDA & training
│
├── model/                          # Saved trained models, scalers, encoders & plots
│   ├── best_model.pkl              # Saved best classification model (Pickle format)
│   ├── best_classification_model.joblib # Saved best classification model (Joblib format)
│   ├── best_regression_model.joblib # Saved best regression model
│   ├── scaler.joblib               # StandardScaler object
│   ├── cat_imputer.joblib          # Categorical Imputer
│   ├── num_imputer.joblib          # Numerical Imputer
│   ├── ohe_encoder.joblib          # OneHotEncoder for nominal categories
│   ├── parent_edu_order.joblib     # Ordinal mapping for parent education
│   ├── family_income_order.joblib   # Ordinal mapping for family income
│   ├── accuracy_comparison.png     # Accuracy bar chart plot
│   ├── confusion_matrix.png        # Confusion matrix plot
│   ├── roc_curve.png               # ROC AUC plot
│   ├── feature_importance.png      # Feature importance plot
│   └── correlation_heatmap.png     # Feature correlation heatmap
│
├── app.py                         # Streamlit Web Application interface
├── train.py                       # ML Training, Benchmarking & Hyperparameter tuning script
├── generate_dataset.py            # Synthetic dataset generator script
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation & line-by-line explanation
```

---

## 📊 Dataset Features & Target

The dataset contains **13 features**:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `Hours Studied` | Continuous (Float) | Number of daily study hours (1.0 - 12.0) |
| `Attendance (%)` | Continuous (Float) | Percentage of classes attended (50.0 - 100.0%) |
| `Previous Exam Score` | Continuous (Float) | Previous test score (0 - 100) |
| `Assignment Score` | Continuous (Float) | Average assignment score (0 - 100) |
| `Sleep Hours` | Continuous (Float) | Nightly sleep duration (4.0 - 10.0 hours) |
| `Internet Access` | Categorical (Binary)| Home internet access (`Yes`, `No`) |
| `Family Income` | Categorical (Ordinal)| Household income (`Low`, `Medium`, `High`) |
| `Parent Education` | Categorical (Ordinal)| Highest parental degree (`High School`, `Bachelor`, `Master`, `Doctorate`) |
| `Study Time` | Continuous (Float) | Total weekly study hours (1.0 - 30.0) |
| `Extra Activities` | Categorical (Binary)| Extracurricular involvement (`Yes`, `No`) |
| `Gender` | Categorical (Binary)| Student gender (`Male`, `Female`) |
| `Age` | Discrete (Integer) | Student age (15 - 25 years) |
| **`Final Exam Score`** | **Target (Continuous)** | **Final exam score out of 100** |
| **`Target_Class`** | **Target (Binary)** | **1 = High Performance (>=70), 0 = Low Performance (<70)** |

---

## ⚡ Quick Start Guide

### 1. Prerequisites & Virtual Environment
Ensure you have Python 3.9+ installed.

```bash
cd Student-Performance-Prediction
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python generate_dataset.py
```

### 3. Run Training & Benchmarking Pipeline
```bash
python train.py
```
This executes data preprocessing, imputes missing values, applies label & one-hot encoding, scales features, trains all 8 ML models, performs hyperparameter tuning via 5-Fold Cross Validation, saves the best model, and generates evaluation plots in `model/`.

### 4. Launch Streamlit Web Application
```bash
streamlit run app.py
```

---

## 🤖 Models & Evaluation Metrics

### 1. Classification Models Evaluated
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVC)
- K-Nearest Neighbors (KNN)
- XGBoost Classifier
- Gradient Boosting Classifier

**Classification Metrics Measured:** Accuracy, Precision, Recall, F1 Score, ROC AUC, 5-Fold Cross Validation Accuracy.

### 2. Regression Models Evaluated
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- Gradient Boosting Regressor

**Regression Metrics Measured:** Mean Absolute Error (MAE), Mean Squared Error (MSE), R² Score.

---

## 📖 Line-by-Line Code Explanations

### A. Explanation of `train.py`

1. `import os, sys, numpy as np, pandas as pd`: Imports standard Python operating system, system tools, numerical arrays (NumPy), and tabular data handling (Pandas).
2. `import matplotlib.pyplot as plt, seaborn as sns, joblib`: Imports Matplotlib and Seaborn for plotting visualizations, and Joblib for saving and loading binary Python objects (like ML models).
3. `from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV`: Imports train/test splitting utilities, cross-validation evaluation tools, and grid search hyperparameter tuning.
4. `from sklearn.impute import SimpleImputer`: Imports `SimpleImputer` to replace missing values using strategies like median or most frequent (mode).
5. `from sklearn.preprocessing import StandardScaler, OneHotEncoder`: Imports `StandardScaler` to scale numerical features to mean=0 and variance=1, and `OneHotEncoder` to transform nominal text categories into binary columns.
6. `from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier`: Imports ensemble tree-based classification models.
7. `from xgboost import XGBClassifier, XGBRegressor`: Imports extreme gradient boosting libraries for fast, scalable tree ensemble learning.
8. `def run_training_pipeline()`: Defines the main executable function carrying out all 9 steps of data engineering, model training, and exporting.
9. `df = pd.read_csv(dataset_path)`: Reads `student_performance.csv` from disk into a Pandas DataFrame.
10. `num_imputer = SimpleImputer(strategy='median')`: Creates an imputer that fills missing numerical values with column medians (robust to outliers).
11. `cat_imputer = SimpleImputer(strategy='most_frequent')`: Creates an imputer that fills missing text categories with the most frequent value.
12. `df['Target_Class'] = (df['Final Exam Score'] >= 70).astype(int)`: Creates binary classification target (`1` for High score >= 70, `0` for Low score < 70).
13. `parent_edu_order = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'Doctorate': 3}`: Maps ordinal parent education strings to logical ascending integers.
14. `family_income_order = {'Low': 0, 'Medium': 1, 'High': 2}`: Maps ordinal family income strings to logical ascending integers.
15. `ohe = OneHotEncoder(drop='first', sparse_output=False)`: Creates a OneHotEncoder for nominal variables, dropping the first category to avoid multicollinearity.
16. `scaler = StandardScaler()`: Scales features so every feature has zero mean and unit variance, preventing large range features from overwhelming models like KNN or SVM.
17. `X_train, X_test, y_train_cls, y_test_cls = train_test_split(...)`: Splits dataset into 80% for training models and 20% for unseen test evaluation, preserving class ratios with `stratify`.
18. `grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=rf_param_grid, cv=5)`: Evaluates all hyperparameter combinations across 5 cross-validation folds to find the optimal Random Forest settings.
19. `joblib.dump(best_clf_model, 'model/best_classification_model.joblib')`: Serializes and saves the best tuned model to disk for production deployment in Streamlit.
20. `plt.savefig(...)`: Saves evaluation plots (Confusion Matrix, ROC Curve, Accuracy Comparison, Correlation Heatmap) as high-resolution PNG images.

---

### B. Explanation of `app.py` (Streamlit Web App)

1. `st.set_page_config(...)`: Sets the browser title, favicon icon, wide screen layout, and sidebar state.
2. `st.markdown("<style>...</style>")`: Inject custom CSS styles for dark-mode glassmorphism cards and custom color-coded metric cards.
3. `@st.cache_resource def load_models_and_preprocessors()`: Caches model loading so the web app runs instantly without re-reading disk files on every user interaction.
4. `hours_studied = st.sidebar.slider(...)`: Renders an interactive sidebar slider allowing the user to select daily study hours.
5. `raw_df = pd.DataFrame([...])`: Wraps user input variables into a single-row Pandas DataFrame matching raw dataset format.
6. `ohe_feat = ohe.transform(...)`: Applies the exact pre-fitted One-Hot Encoder to transform user text choices into binary dummy variables.
7. `input_scaled = scaler.transform(processed_df)`: Normalizes user inputs using the pre-fitted `StandardScaler`.
8. `pred_class = clf_model.predict(input_scaled)[0]`: Obtains class prediction (1 = High Performance, 0 = Low Performance).
9. `pred_prob = clf_model.predict_proba(input_scaled)[0][1]`: Computes exact probability percentage (0-100%) of the student achieving high performance.
10. `pred_score = reg_model.predict(input_scaled)[0]`: Predicts the exact estimated numerical final exam score (out of 100).
11. `st.progress(pred_prob)`: Visualizes the high performance probability as an animated progress bar gauge.
12. `st.image(...)`: Renders generated model evaluation plots in an interactive tab view.
13. `recommendations.append(...)`: Generates personalized AI advice based on student metric thresholds (e.g., advising attendance or sleep habit increases).

---

## 🎯 Summary of Accomplished Tasks

- [x] Project Name & Objective setup
- [x] Synthetic dataset generation with 13 features
- [x] Missing value imputation with `SimpleImputer`
- [x] Label Encoding & One-Hot Encoding
- [x] Feature Scaling using `StandardScaler`
- [x] Exploratory Data Analysis & Correlation Heatmap
- [x] Multi-model training (8 models: Linear, Logistic, Tree, RF, SVM, KNN, XGBoost, Gradient Boosting)
- [x] Evaluation across 8 metrics (Accuracy, Precision, Recall, F1, ROC AUC, MAE, MSE, R²)
- [x] 5-Fold Stratified Cross Validation & GridSearchCV Hyperparameter Tuning
- [x] Visual plots saved: Confusion Matrix, ROC Curve, Accuracy Comparison, Feature Importance
- [x] Model serialization using `joblib` & `pickle`
- [x] Streamlit Web Application with interactive dashboard & recommendations
- [x] Line-by-line simple language explanations
