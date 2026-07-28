import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, mean_absolute_error, mean_squared_error, r2_score
)

# XGBoost (Optional / gracefully handled if libomp is missing on macOS)
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except Exception as e:
    HAS_XGBOOST = False
    print(f"Note: XGBoost library could not be loaded ({e}). Continuing with Scikit-Learn Gradient Boosting.")

def run_training_pipeline():
    print("=" * 70)
    print("      STUDENT PERFORMANCE PREDICTION - MACHINE LEARNING PIPELINE      ")
    print("=" * 70)
    
    # Define directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'dataset', 'student_performance.csv')
    model_dir = os.path.join(base_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # STEP 1: LOAD DATASET
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Loading Dataset...")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}. Please run generate_dataset.py first.")
        sys.exit(1)
        
    df = pd.read_csv(dataset_path)
    print(f"Dataset Loaded Successfully! Shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # -------------------------------------------------------------------------
    # STEP 2: DATA PREPROCESSING & HANDLING MISSING VALUES
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Preprocessing & Handling Missing Values...")
    print("Missing values per column before imputation:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    # Separate numeric and categorical columns
    numeric_cols = ['Hours Studied', 'Attendance (%)', 'Previous Exam Score', 'Assignment Score', 'Sleep Hours', 'Study Time', 'Age']
    categorical_cols = ['Internet Access', 'Family Income', 'Parent Education', 'Extra Activities', 'Gender']
    
    # Impute Missing Values
    num_imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    print("Missing values after imputation:", df.isnull().sum().sum())
    
    # Define Binary Classification Target (1 = High Performance [>=70], 0 = Low Performance [<70])
    df['Target_Class'] = (df['Final Exam Score'] >= 70).astype(int)
    print(f"\nTarget Class Distribution:\n{df['Target_Class'].value_counts(normalize=True).round(3) * 100}%")
    
    # Save Imputers
    joblib.dump(num_imputer, os.path.join(model_dir, 'num_imputer.joblib'))
    joblib.dump(cat_imputer, os.path.join(model_dir, 'cat_imputer.joblib'))
    
    # -------------------------------------------------------------------------
    # STEP 3: ENCODING CATEGORICAL FEATURES & SCALING
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Encoding Features & Scaling...")
    df_encoded = df.copy()
    
    # Ordinal Label Encoding for ordered categories
    parent_edu_order = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'Doctorate': 3}
    family_income_order = {'Low': 0, 'Medium': 1, 'High': 2}
    
    df_encoded['Parent Education'] = df_encoded['Parent Education'].map(parent_edu_order)
    df_encoded['Family Income'] = df_encoded['Family Income'].map(family_income_order)
    
    # One-Hot Encoding for nominal binary/multi-class categories
    nominal_cols = ['Internet Access', 'Extra Activities', 'Gender']
    ohe = OneHotEncoder(drop='first', sparse_output=False)
    ohe_features = ohe.fit_transform(df_encoded[nominal_cols])
    ohe_feature_names = ohe.get_feature_names_out(nominal_cols)
    
    df_ohe = pd.DataFrame(ohe_features, columns=ohe_feature_names, index=df_encoded.index)
    
    # Combine engineered features
    df_final = pd.concat([
        df_encoded[numeric_cols + ['Parent Education', 'Family Income']],
        df_ohe,
        df_encoded[['Final Exam Score', 'Target_Class']]
    ], axis=1)
    
    # Feature & Target Matrices
    feature_cols = [c for c in df_final.columns if c not in ['Final Exam Score', 'Target_Class']]
    X = df_final[feature_cols]
    y_class = df_final['Target_Class']
    y_reg = df_final['Final Exam Score']
    
    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
    
    # Save Encoders & Scaler
    joblib.dump(parent_edu_order, os.path.join(model_dir, 'parent_edu_order.joblib'))
    joblib.dump(family_income_order, os.path.join(model_dir, 'family_income_order.joblib'))
    joblib.dump(ohe, os.path.join(model_dir, 'ohe_encoder.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    joblib.dump(feature_cols, os.path.join(model_dir, 'feature_cols.joblib'))
    
    # -------------------------------------------------------------------------
    # STEP 4: EXPLORATORY DATA ANALYSIS & CORRELATION HEATMAP
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Generating Correlation Heatmap...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_final.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap - Student Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    print("Correlation heatmap saved to model/correlation_heatmap.png")
    
    # -------------------------------------------------------------------------
    # STEP 5: TRAIN-TEST SPLIT
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Splitting Data into Train & Test sets (80-20 Split)...")
    X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg = train_test_split(
        X_scaled, y_class, y_reg, test_size=0.20, random_state=42, stratify=y_class
    )
    print(f"Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")
    
    # -------------------------------------------------------------------------
    # STEP 6: MODEL TRAINING & EVALUATION (CLASSIFICATION & REGRESSION)
    # -------------------------------------------------------------------------
    print("\n[STEP 6] Training & Benchmarking Classification Models...")
    
    classification_models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    if HAS_XGBOOST:
        classification_models['XGBoost'] = XGBClassifier(random_state=42, eval_metric='logloss')

    
    clf_results = []
    
    for name, model in classification_models.items():
        model.fit(X_train, y_train_cls)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        acc = accuracy_score(y_test_cls, y_pred)
        prec = precision_score(y_test_cls, y_pred)
        rec = recall_score(y_test_cls, y_pred)
        f1 = f1_score(y_test_cls, y_pred)
        roc_auc = roc_auc_score(y_test_cls, y_prob)
        
        cv_scores = cross_val_score(model, X_train, y_train_cls, cv=5, scoring='accuracy')
        
        clf_results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'ROC AUC': roc_auc,
            '5-Fold CV Mean Acc': cv_scores.mean()
        })
        
    clf_df = pd.DataFrame(clf_results).sort_values(by='F1 Score', ascending=False)
    print("\nClassification Model Benchmark Comparison:")
    print(clf_df.to_string(index=False))
    
    # -------------------------------------------------------------------------
    # STEP 7: REGRESSION BENCHMARK (Continuous Score Prediction)
    # -------------------------------------------------------------------------
    print("\n[STEP 7] Training Regression Models for Score Prediction...")
    regression_models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=42, max_depth=5),
        'Random Forest Regressor': RandomForestRegressor(random_state=42, n_estimators=100),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42)
    }
    if HAS_XGBOOST:
        regression_models['XGBoost Regressor'] = XGBRegressor(random_state=42)

    
    reg_results = []
    for name, model in regression_models.items():
        model.fit(X_train, y_train_reg)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test_reg, y_pred)
        mse = mean_squared_error(y_test_reg, y_pred)
        r2 = r2_score(y_test_reg, y_pred)
        
        reg_results.append({
            'Model': name,
            'MAE': mae,
            'MSE': mse,
            'R² Score': r2
        })
        
    reg_df = pd.DataFrame(reg_results).sort_values(by='R² Score', ascending=False)
    print("\nRegression Model Benchmark Comparison:")
    print(reg_df.to_string(index=False))
    
    # Save best regression model
    best_reg_name = reg_df.iloc[0]['Model']
    best_reg_model = regression_models[best_reg_name]
    joblib.dump(best_reg_model, os.path.join(model_dir, 'best_regression_model.joblib'))
    print(f"\nBest Regression Model: {best_reg_name} (R²: {reg_df.iloc[0]['R² Score']:.4f})")
    
    # -------------------------------------------------------------------------
    # STEP 8: HYPERPARAMETER TUNING (Best Classifier - Random Forest / XGBoost)
    # -------------------------------------------------------------------------
    print("\n[STEP 8] Performing Hyperparameter Tuning for Random Forest Classifier...")
    rf_param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'criterion': ['gini', 'entropy']
    }
    
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=rf_param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train_cls)
    
    best_clf_model = grid_search.best_estimator_
    print(f"Best Parameters found: {grid_search.best_params_}")
    
    # Final evaluation of tuned best model
    y_pred_best = best_clf_model.predict(X_test)
    y_prob_best = best_clf_model.predict_proba(X_test)[:, 1]
    
    final_acc = accuracy_score(y_test_cls, y_pred_best)
    final_f1 = f1_score(y_test_cls, y_pred_best)
    print(f"Tuned Best Model Test Accuracy: {final_acc:.4f} | Test F1 Score: {final_f1:.4f}")
    
    # Save Best Classification Model
    joblib.dump(best_clf_model, os.path.join(model_dir, 'best_model.pkl'))
    joblib.dump(best_clf_model, os.path.join(model_dir, 'best_classification_model.joblib'))
    print("Saved best classification model to model/best_model.pkl & model/best_classification_model.joblib")
    
    # -------------------------------------------------------------------------
    # STEP 9: GENERATE EVALUATION PLOTS
    # -------------------------------------------------------------------------
    print("\n[STEP 9] Generating & Saving Evaluation Visualizations...")
    
    # 1. Model Accuracy Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=clf_df, x='Model', y='Accuracy', palette='viridis')
    plt.title('Classification Accuracy Comparison across ML Models', fontsize=14, fontweight='bold')
    plt.xticks(rotation=30)
    plt.ylim(0.7, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'accuracy_comparison.png'), dpi=300)
    plt.close()
    
    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y_test_cls, y_pred_best)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Score (<70)', 'High Score (>=70)'], yticklabels=['Low Score (<70)', 'High Score (>=70)'])
    plt.title('Confusion Matrix - Best Tuned Model', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # 3. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test_cls, y_prob_best)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc_score(y_test_cls, y_prob_best):.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'roc_curve.png'), dpi=300)
    plt.close()
    
    # 4. Feature Importance Plot
    if hasattr(best_clf_model, 'feature_importances_'):
        importances = best_clf_model.feature_importances_
        feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_df, x='Importance', y='Feature', palette='plasma')
        plt.title('Feature Importances - Student Performance Prediction', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'feature_importance.png'), dpi=300)
        plt.close()
        
    print("All Evaluation Plots saved to model/ directory!")
    print("\n" + "=" * 70)
    print("             TRAINING PIPELINE COMPLETED SUCCESSFULLY!             ")
    print("=" * 70)

if __name__ == '__main__':
    run_training_pipeline()
