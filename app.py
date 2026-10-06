import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Auto-launch Streamlit server if executed directly via `python app.py` (e.g. on Render)
if __name__ == "__main__" and get_script_run_ctx() is None:
    from streamlit.web import cli as stcli
    port = os.environ.get("PORT", "8501")
    sys.argv = ["streamlit", "run", sys.argv[0], "--server.port", str(port), "--server.address", "0.0.0.0"]
    sys.exit(stcli.main())

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance AI Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)
 

# Custom CSS for glassmorphism aesthetics and dark mode theme
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Header Card */
    .header-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 20px;
        border-left: 5px solid #6366f1;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    .high-perf {
        border-left-color: #22c55e !important;
        background: rgba(34, 197, 94, 0.1);
    }
    
    .low-perf {
        border-left-color: #ef4444 !important;
        background: rgba(239, 68, 68, 0.1);
    }
    
    /* Recommendation Card */
    .rec-card {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 16px;
        margin-top: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. LOAD TRAINED ARTIFACTS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models_and_preprocessors():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'model')
    
    try:
        clf_model = joblib.load(os.path.join(model_dir, 'best_classification_model.joblib'))
        reg_model = joblib.load(os.path.join(model_dir, 'best_regression_model.joblib'))
        num_imputer = joblib.load(os.path.join(model_dir, 'num_imputer.joblib'))
        cat_imputer = joblib.load(os.path.join(model_dir, 'cat_imputer.joblib'))
        parent_edu_order = joblib.load(os.path.join(model_dir, 'parent_edu_order.joblib'))
        family_income_order = joblib.load(os.path.join(model_dir, 'family_income_order.joblib'))
        ohe = joblib.load(os.path.join(model_dir, 'ohe_encoder.joblib'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        feature_cols = joblib.load(os.path.join(model_dir, 'feature_cols.joblib'))
        return clf_model, reg_model, num_imputer, cat_imputer, parent_edu_order, family_income_order, ohe, scaler, feature_cols
    except Exception as e:
        st.error(f"Error loading models or preprocessors. Please run `python train.py` first! Details: {e}")
        return None, None, None, None, None, None, None, None, None

clf_model, reg_model, num_imputer, cat_imputer, parent_edu_order, family_income_order, ohe, scaler, feature_cols = load_models_and_preprocessors()

# -----------------------------------------------------------------------------
# 3. HEADER & TITLE
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="header-card">
        <h1 style="color: #6366f1; margin-bottom: 8px;">🎓 Student Performance Predictor</h1>
        <p style="font-size: 1.1rem; color: #cbd5e1;">
            Empowering educators and students with Machine Learning to forecast academic outcomes, analyze risk factors, and maximize potential.
        </p>
    </div>
""", unsafe_allow_html=True)

if clf_model is None:
    st.warning("⚠️ Trained models not found in `model/` directory. Please run `python train.py` in your terminal to train and export the models.")
    st.stop()

# -----------------------------------------------------------------------------
# 4. SIDEBAR - STUDENT FEATURE INPUTS
# -----------------------------------------------------------------------------
st.sidebar.header("📋 Student Profile Inputs")

with st.sidebar.form(key="student_form"):
    st.subheader("📚 Academic Metrics")
    hours_studied = st.slider("Hours Studied per Day", min_value=1.0, max_value=12.0, value=6.0, step=0.5)
    attendance = st.slider("Attendance (%)", min_value=50.0, max_value=100.0, value=85.0, step=1.0)
    previous_score = st.number_input("Previous Exam Score (0-100)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
    assignment_score = st.number_input("Assignment Score (0-100)", min_value=0.0, max_value=100.0, value=78.0, step=1.0)
    study_time = st.slider("Weekly Study Time (Hours)", min_value=1.0, max_value=30.0, value=15.0, step=1.0)

    st.subheader("👤 Personal & Lifestyle Factors")
    sleep_hours = st.slider("Sleep Hours per Night", min_value=4.0, max_value=10.0, value=7.0, step=0.5)
    age = st.number_input("Age", min_value=15, max_value=25, value=18, step=1)
    gender = st.selectbox("Gender", options=['Male', 'Female'])
    internet_access = st.selectbox("Internet Access at Home", options=['Yes', 'No'])
    extra_activities = st.selectbox("Extracurricular Activities", options=['Yes', 'No'])

    st.subheader("🏡 Family & Socioeconomic")
    family_income = st.selectbox("Family Income Level", options=['Low', 'Medium', 'High'], index=1)
    parent_education = st.selectbox("Parent Education Level", options=['High School', 'Bachelor', 'Master', 'Doctorate'], index=1)

    submit_button = st.form_submit_button(label="🔮 Predict Student Performance")

# -----------------------------------------------------------------------------
# 5. PREPROCESSING & PREDICTION LOGIC
# -----------------------------------------------------------------------------
def preprocess_input(hours_studied, attendance, previous_score, assignment_score, sleep_hours, study_time, age,
                     gender, internet_access, extra_activities, family_income, parent_education):
    # Construct raw DataFrame matching dataset features
    raw_df = pd.DataFrame([{
        'Hours Studied': hours_studied,
        'Attendance (%)': attendance,
        'Previous Exam Score': previous_score,
        'Assignment Score': assignment_score,
        'Sleep Hours': sleep_hours,
        'Study Time': study_time,
        'Age': age,
        'Internet Access': internet_access,
        'Family Income': family_income,
        'Parent Education': parent_education,
        'Extra Activities': extra_activities,
        'Gender': gender
    }])

    # Ordinal Mapping
    raw_df['Parent Education'] = raw_df['Parent Education'].map(parent_edu_order)
    raw_df['Family Income'] = raw_df['Family Income'].map(family_income_order)

    # One-Hot Encoding
    nominal_cols = ['Internet Access', 'Extra Activities', 'Gender']
    ohe_feat = ohe.transform(raw_df[nominal_cols])
    ohe_df = pd.DataFrame(ohe_feat, columns=ohe.get_feature_names_out(nominal_cols))

    numeric_cols = ['Hours Studied', 'Attendance (%)', 'Previous Exam Score', 'Assignment Score', 'Sleep Hours', 'Study Time', 'Age']
    
    processed_df = pd.concat([
        raw_df[numeric_cols + ['Parent Education', 'Family Income']],
        ohe_df
    ], axis=1)

    # Reorder columns exactly as in training
    processed_df = processed_df[feature_cols]

    # Scaling
    scaled_features = pd.DataFrame(scaler.transform(processed_df), columns=feature_cols)
    return scaled_features


# -----------------------------------------------------------------------------
# 6. MAIN CONTENT & TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🎯 Live Prediction", "📊 Model Performance & EDA", "💡 AI Recommendations"])

with tab1:
    st.subheader("Results Dashboard")
    
    # Preprocess & Predict
    input_scaled = preprocess_input(
        hours_studied, attendance, previous_score, assignment_score, sleep_hours, study_time, age,
        gender, internet_access, extra_activities, family_income, parent_education
    )
    
    pred_class = clf_model.predict(input_scaled)[0]
    pred_prob = clf_model.predict_proba(input_scaled)[0][1] if hasattr(clf_model, "predict_proba") else (1.0 if pred_class == 1 else 0.0)
    pred_score = reg_model.predict(input_scaled)[0]

    # Layout Columns for Results
    col1, col2, col3 = st.columns(3)

    with col1:
        if pred_class == 1:
            st.markdown(f"""
                <div class="metric-card high-perf">
                    <h3 style="color: #22c55e; margin:0;">🌟 High Performance</h3>
                    <p style="font-size: 1.8rem; font-weight: bold; margin: 8px 0 0 0;">PASS / EXCEL</p>
                    <small>Predicted Target Class: 1 (High Score >= 70%)</small>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="metric-card low-perf">
                    <h3 style="color: #ef4444; margin:0;">⚠️ Low Performance</h3>
                    <p style="font-size: 1.8rem; font-weight: bold; margin: 8px 0 0 0;">AT RISK / LOW</p>
                    <small>Predicted Target Class: 0 (Score < 70%)</small>
                </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #6366f1; margin:0;">📈 High Performance Prob.</h3>
                <p style="font-size: 2rem; font-weight: bold; margin: 8px 0 0 0; color: #f8fafc;">{pred_prob * 100:.1f}%</p>
                <small>Confidence Score from Best Ensemble Model</small>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #38bdf8; margin:0;">📝 Estimated Final Score</h3>
                <p style="font-size: 2rem; font-weight: bold; margin: 8px 0 0 0; color: #f8fafc;">{pred_score:.1f} / 100</p>
                <small>Continuous Regression Model Output</small>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(float(np.clip(pred_prob, 0.0, 1.0)), text=f"High Performance Probability Gauge: {pred_prob*100:.1f}%")

with tab2:
    st.subheader("Model Visualizations & EDA Insights")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'model')

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Model Accuracy Comparison**")
        acc_path = os.path.join(model_dir, 'accuracy_comparison.png')
        if os.path.exists(acc_path):
            st.image(acc_path, width="stretch")
        
        st.write("**Confusion Matrix**")
        cm_path = os.path.join(model_dir, 'confusion_matrix.png')
        if os.path.exists(cm_path):
            st.image(cm_path, width="stretch")

    with col_b:
        st.write("**Feature Importance Ranking**")
        fi_path = os.path.join(model_dir, 'feature_importance.png')
        if os.path.exists(fi_path):
            st.image(fi_path, width="stretch")

        st.write("**ROC Curve**")
        roc_path = os.path.join(model_dir, 'roc_curve.png')
        if os.path.exists(roc_path):
            st.image(roc_path, width="stretch")

    st.write("**Feature Correlation Heatmap**")
    corr_path = os.path.join(model_dir, 'correlation_heatmap.png')
    if os.path.exists(corr_path):
        st.image(corr_path, width="stretch")


with tab3:
    st.subheader("Personalized Action Plan for Student Improvement")
    
    recommendations = []
    if attendance < 80.0:
        recommendations.append("🚩 **Attendance Alert**: Attendance is currently below 80%. Increasing class attendance to 85%+ has the strongest statistical correlation with higher final exam scores.")
    if hours_studied < 5.0:
        recommendations.append("📚 **Study Habit**: Daily study hours are under 5 hours. Increasing study time by just 1.5 - 2 hours daily can significantly boost retention and exam performance.")
    if sleep_hours < 7.0:
        recommendations.append("😴 **Sleep Schedule**: Getting less than 7 hours of sleep negatively impacts cognitive performance and exam recall. Aim for 7.5 - 8 hours per night.")
    if assignment_score < 70.0:
        recommendations.append("📝 **Assignment Focus**: Assignment scores reflect ongoing subject mastery. Seeking tutoring or peer study groups for assignments will build core knowledge.")

    if not recommendations:
        st.success("🎉 **Outstanding Habits!** The student maintains excellent attendance, study hours, sleep schedule, and academic metrics. Keep up the high standard!")
    else:
        for rec in recommendations:
            st.markdown(f'<div class="rec-card">{rec}</div>', unsafe_allow_html=True)
