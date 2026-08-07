import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction")
st.write("Predict a student's exam score using Machine Learning.")

st.header("Enter Student Details")

# -----------------------------
# Numeric Inputs
# -----------------------------

hours = st.number_input("Hours Studied", 0.0, 24.0, 5.0)

attendance = st.number_input("Attendance (%)", 0.0, 100.0, 80.0)

sleep = st.number_input("Sleep Hours", 0.0, 12.0, 7.0)

previous = st.number_input("Previous Scores", 0.0, 100.0, 70.0)

tutoring = st.number_input("Tutoring Sessions", 0, 20, 2)

physical = st.number_input("Physical Activity (hours/week)", 0.0, 20.0, 4.0)

study_efficiency = st.number_input("Study Efficiency", 0.0, 100.0, 60.0)

academic_consistency = st.number_input("Academic Consistency", 0.0, 100.0, 60.0)

healthy = st.number_input("Healthy Lifestyle Index", 0.0, 100.0, 60.0)

# -----------------------------
# Dropdown Inputs
# -----------------------------

parental = st.selectbox(
    "Parental Involvement",
    list(label_encoders["Parental_Involvement"].classes_)
)

resources = st.selectbox(
    "Access to Resources",
    list(label_encoders["Access_to_Resources"].classes_)
)

extra = st.selectbox(
    "Extracurricular Activities",
    list(label_encoders["Extracurricular_Activities"].classes_)
)

motivation = st.selectbox(
    "Motivation Level",
    list(label_encoders["Motivation_Level"].classes_)
)

internet = st.selectbox(
    "Internet Access",
    list(label_encoders["Internet_Access"].classes_)
)

income = st.selectbox(
    "Family Income",
    list(label_encoders["Family_Income"].classes_)
)

teacher = st.selectbox(
    "Teacher Quality",
    list(label_encoders["Teacher_Quality"].classes_)
)

school = st.selectbox(
    "School Type",
    list(label_encoders["School_Type"].classes_)
)

peer = st.selectbox(
    "Peer Influence",
    list(label_encoders["Peer_Influence"].classes_)
)

learning = st.selectbox(
    "Learning Disabilities",
    list(label_encoders["Learning_Disabilities"].classes_)
)

parent_edu = st.selectbox(
    "Parental Education Level",
    list(label_encoders["Parental_Education_Level"].classes_)
)

distance = st.selectbox(
    "Distance From Home",
    list(label_encoders["Distance_from_Home"].classes_)
)

gender = st.selectbox(
    "Gender",
    list(label_encoders["Gender"].classes_)
)

# -----------------------------
# Predict Button
# -----------------------------

if st.button("Predict Score"):

    input_df = pd.DataFrame({

        "Hours_Studied":[hours],
        "Attendance":[attendance],
        "Parental_Involvement":[label_encoders["Parental_Involvement"].transform([parental])[0]],
        "Access_to_Resources":[label_encoders["Access_to_Resources"].transform([resources])[0]],
        "Extracurricular_Activities":[label_encoders["Extracurricular_Activities"].transform([extra])[0]],
        "Sleep_Hours":[sleep],
        "Previous_Scores":[previous],
        "Motivation_Level":[label_encoders["Motivation_Level"].transform([motivation])[0]],
        "Internet_Access":[label_encoders["Internet_Access"].transform([internet])[0]],
        "Tutoring_Sessions":[tutoring],
        "Family_Income":[label_encoders["Family_Income"].transform([income])[0]],
        "Teacher_Quality":[label_encoders["Teacher_Quality"].transform([teacher])[0]],
        "School_Type":[label_encoders["School_Type"].transform([school])[0]],
        "Peer_Influence":[label_encoders["Peer_Influence"].transform([peer])[0]],
        "Physical_Activity":[physical],
        "Learning_Disabilities":[label_encoders["Learning_Disabilities"].transform([learning])[0]],
        "Parental_Education_Level":[label_encoders["Parental_Education_Level"].transform([parent_edu])[0]],
        "Distance_from_Home":[label_encoders["Distance_from_Home"].transform([distance])[0]],
        "Gender":[label_encoders["Gender"].transform([gender])[0]],
        "Study_Efficiency":[study_efficiency],
        "Academic_Consistency":[academic_consistency],
        "Healthy_Lifestyle_Index":[healthy]

    })

    scaled = scaler.transform(input_df)

    prediction = model.predict(scaled)

    st.success(f"🎯 Predicted Exam Score: {prediction[0]:.2f}")

st.sidebar.title("About")
st.sidebar.info(
    """
    Student Performance Prediction

    Built using:
    - Python
    - Streamlit
    - Scikit-learn
    - Machine Learning
    """
)