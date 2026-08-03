import joblib
import pandas as pd

# Load saved files
model = joblib.load("C:\\Users\\pushp\\OneDrive\\Desktop\\AI &ML project\\student-score-prediction-1\\models\\best_model.pkl")
scaler = joblib.load("C:\\Users\\pushp\\OneDrive\\Desktop\\AI &ML project\\student-score-prediction-1\\models\\scaler.pkl")
label_encoders = joblib.load("C:\\Users\\pushp\\OneDrive\\Desktop\\AI &ML project\\student-score-prediction-1\\models\\label_encoders.pkl")

print("="*60)
print(" STUDENT PERFORMANCE PREDICTION SYSTEM ")
print("="*60)

# ---------- Numerical Inputs ----------

hours = float(input("Hours Studied: "))
attendance = float(input("Attendance: "))
sleep = float(input("Sleep Hours: "))
previous = float(input("Previous Scores: "))
tutoring = float(input("Tutoring Sessions: "))
physical = float(input("Physical Activity : "))
study_efficiency = float(input("Study Efficiency: "))
academic_consistency = float(input("Academic Consistency: "))
healthy_index = float(input("Healthy Lifestyle Index: "))

# ---------- Categorical Inputs ----------

parental = input("Parental Involvement (Low/Medium/High): ").title()

resources = input("Access to Resources (Low/Medium/High): ").title()
extra = input("Extracurricular Activities (Yes/No): ").title()
motivation = input("Motivation Level (Low/Medium/High): ").title()
internet = input("Internet Access (Yes/No): ").title()
income = input("Family Income (Low/Medium/High): ").title()
teacher = input("Teacher Quality (Low/Medium/High): ").title()
school = input("School Type (Public/Private): ").title()
peer = input("Peer Influence (Negative/Neutral/Positive): ").title()
learning = input("Learning Disabilities (Yes/No): ").title()
parent_edu = input("Parental Education Level (High School/College/Postgraduate): ").title()
distance = input("Distance from Home (Near/Moderate/Far): ").title()
gender = input("Gender (Male/Female): ").title()

# Encode categorical features
parental = label_encoders["Parental_Involvement"].transform([parental])[0]
resources = label_encoders["Access_to_Resources"].transform([resources])[0]
extra = label_encoders["Extracurricular_Activities"].transform([extra])[0]
motivation = label_encoders["Motivation_Level"].transform([motivation])[0]
internet = label_encoders["Internet_Access"].transform([internet])[0]
income = label_encoders["Family_Income"].transform([income])[0]
teacher = label_encoders["Teacher_Quality"].transform([teacher])[0]
school = label_encoders["School_Type"].transform([school])[0]
peer = label_encoders["Peer_Influence"].transform([peer])[0]
learning = label_encoders["Learning_Disabilities"].transform([learning])[0]
parent_edu = label_encoders["Parental_Education_Level"].transform([parent_edu])[0]
distance = label_encoders["Distance_from_Home"].transform([distance])[0]
gender = label_encoders["Gender"].transform([gender])[0]

# Create DataFrame
input_df = pd.DataFrame({

"Hours_Studied":[hours],
"Attendance":[attendance],
"Parental_Involvement":[parental],
"Access_to_Resources":[resources],
"Extracurricular_Activities":[extra],
"Sleep_Hours":[sleep],
"Previous_Scores":[previous],
"Motivation_Level":[motivation],
"Internet_Access":[internet],
"Tutoring_Sessions":[tutoring],
"Family_Income":[income],
"Teacher_Quality":[teacher],
"School_Type":[school],
"Peer_Influence":[peer],
"Physical_Activity":[physical],
"Learning_Disabilities":[learning],
"Parental_Education_Level":[parent_edu],
"Distance_from_Home":[distance],
"Gender":[gender],
"Study_Efficiency":[study_efficiency],
"Academic_Consistency":[academic_consistency],
"Healthy_Lifestyle_Index":[healthy_index]

})

# Scale data
scaled = scaler.transform(input_df)

# Prediction
prediction = model.predict(scaled)

print("\n"+"="*60)
print(f"Predicted Exam Score : {prediction[0]:.2f}")
print("="*60)

