import os
import numpy as np
import pandas as pd

def generate_student_dataset(num_samples=1000, seed=42):
    np.random.seed(seed)
    
    # 1. Feature Generation
    hours_studied = np.round(np.random.uniform(1, 10, num_samples), 1)
    attendance = np.round(np.random.uniform(50, 100, num_samples), 1)
    previous_score = np.round(np.random.normal(70, 15, num_samples), 1)
    previous_score = np.clip(previous_score, 30, 100)
    
    assignment_score = np.round(np.random.normal(75, 12, num_samples), 1)
    assignment_score = np.clip(assignment_score, 35, 100)
    
    sleep_hours = np.round(np.random.normal(7, 1.2, num_samples), 1)
    sleep_hours = np.clip(sleep_hours, 4, 10)
    
    internet_access = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.85, 0.15])
    family_income = np.random.choice(['Low', 'Medium', 'High'], size=num_samples, p=[0.3, 0.5, 0.2])
    parent_education = np.random.choice(['High School', 'Bachelor', 'Master', 'Doctorate'], size=num_samples, p=[0.4, 0.35, 0.2, 0.05])
    
    study_time = np.round(hours_studied * 2.5 + np.random.normal(0, 2, num_samples), 1)
    study_time = np.clip(study_time, 1, 30)
    
    extra_activities = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.45, 0.55])
    gender = np.random.choice(['Male', 'Female'], size=num_samples, p=[0.5, 0.5])
    age = np.random.randint(15, 23, size=num_samples)
    
    # Encoding map for numerical target synthesis
    parent_edu_map = {'High School': 0, 'Bachelor': 2, 'Master': 4, 'Doctorate': 6}
    income_map = {'Low': 0, 'Medium': 2, 'High': 4}
    internet_map = {'Yes': 3, 'No': 0}
    extra_map = {'Yes': 2, 'No': 0}
    
    parent_edu_val = np.array([parent_edu_map[e] for e in parent_education])
    income_val = np.array([income_map[i] for i in family_income])
    internet_val = np.array([internet_map[i] for i in internet_access])
    extra_val = np.array([extra_map[e] for e in extra_activities])
    
    # 2. Final Exam Score Calculation with non-linear relationships & noise
    score_raw = (
        0.35 * previous_score +
        0.25 * assignment_score +
        0.25 * (attendance * 0.4) +
        1.8 * hours_studied +
        0.4 * parent_edu_val +
        0.5 * income_val +
        0.6 * internet_val +
        0.5 * extra_val +
        0.5 * (sleep_hours - 6) +
        np.random.normal(0, 4, num_samples)
    )
    
    # Normalize score to standard 0-100 range
    final_score = np.round(np.clip(score_raw, 30, 100), 1)
    
    # 3. Create DataFrame
    df = pd.DataFrame({
        'Hours Studied': hours_studied,
        'Attendance (%)': attendance,
        'Previous Exam Score': previous_score,
        'Assignment Score': assignment_score,
        'Sleep Hours': sleep_hours,
        'Internet Access': internet_access,
        'Family Income': family_income,
        'Parent Education': parent_education,
        'Study Time': study_time,
        'Extra Activities': extra_activities,
        'Gender': gender,
        'Age': age,
        'Final Exam Score': final_score
    })
    
    # Introduce small amount (~2%) missing values to demonstrate missing value handling
    for col in ['Hours Studied', 'Attendance (%)', 'Previous Exam Score', 'Family Income']:
        mask = np.random.rand(len(df)) < 0.02
        df.loc[mask, col] = np.nan
        
    return df

if __name__ == '__main__':
    output_dir = '/Users/nikhilbishnoi/.gemini/antigravity/scratch/Student-Performance-Prediction/dataset'
    os.makedirs(output_dir, exist_ok=True)
    df = generate_student_dataset()
    filepath = os.path.join(output_dir, 'student_performance.csv')
    df.to_csv(filepath, index=False)
    print(f"Dataset generated successfully at: {filepath}")
    print(f"Shape: {df.shape}")
    print(df.head())
