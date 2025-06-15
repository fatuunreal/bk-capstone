import streamlit as st
import pandas as pd
import numpy as np
import pickle

# === Load Models ===
with open("rf_default.sav", "rb") as f: rf_default = pickle.load(f)
with open("rf_tuned.sav", "rb") as f: rf_tuned = pickle.load(f)
with open("svm_default.sav", "rb") as f: svm_default = pickle.load(f)
with open("svm_tuned.sav", "rb") as f: svm_tuned = pickle.load(f)
with open("logreg_default.sav", "rb") as f: logreg_default = pickle.load(f)
with open("logreg_tuned.sav", "rb") as f: logreg_tuned = pickle.load(f)

# === Load Scaler, Feature Order, dan Label Encoders ===
with open("scaler.pkl", "rb") as f: scaler = pickle.load(f)
with open("feature_columns.pkl", "rb") as f: feature_order = pickle.load(f)
with open("label_encoders.pkl", "rb") as f: label_encoders = pickle.load(f)

# === Label Mapping === (sesuai urutan LabelEncoder y)
label_mapping = {
    0: 'Insufficient_Weight',
    1: 'Normal_Weight',
    2: 'Obesity_Type_I',
    3: 'Obesity_Type_II',
    4: 'Obesity_Type_III',
    5: 'Overweight_Level_I',
    6: 'Overweight_Level_II',
}

# === Streamlit App ===
st.title("Obesity Classification App")
st.write("Masukkan data pribadi dan kebiasaan Anda:")

# === Input Form ===
age = st.slider("Age", 10, 100, 25)
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.7)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
calc = st.selectbox("Alcohol Consumption (CALC)", ["no", "Sometimes", "Frequently", "Always"])
favc = st.selectbox("Frequent High Calorie Food Consumption (FAVC)", ["yes", "no"])
fcvc = st.slider("Vegetable Intake Frequency (FCVC)", 1, 3, 2)
ncp = st.slider("Number of Meals per Day (NCP)", 1, 5, 3)
scc = st.selectbox("Calorie Monitoring (SCC)", ["yes", "no"])
smoke = st.selectbox("Do you smoke? (SMOKE)", ["yes", "no"])
ch2o = st.slider("Water Intake (CH2O)", 1, 3, 2)
famhist = st.selectbox("Family History of Overweight", ["yes", "no"])
faf = st.slider("Physical Activity Frequency (FAF)", 0, 4, 1)
tue = st.slider("Technology Use Time (TUE) (hrs/day)", 0, 5, 2)
caec = st.selectbox("Eating Between Meals (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Transportation Mode", ["Public_Transportation", "Walking", "Motorbike", "Bike", "Automobile"])

if st.button("Klasifikasi Obesitas"):
    # Buat dictionary input user
    input_dict = {
        'Age': age,
        'Height': height,
        'Weight': weight,
        'FCVC': fcvc,
        'NCP': ncp,
        'CH2O': ch2o,
        'FAF': faf,
        'TUE': tue,
        'Gender': gender,
        'family_history_with_overweight': famhist,
        'FAVC': favc,
        'SCC': scc,
        'SMOKE': smoke,
        'CALC': calc,
        'CAEC': caec,
        'MTRANS': mtrans
    }

    input_df = pd.DataFrame([input_dict])

    # Terapkan label encoding sesuai yang digunakan saat training
    for col, encoder in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    # Reorder kolom sesuai training
    input_df = input_df[feature_order]
    X_scaled = scaler.transform(input_df)

    # Model prediksi
    models = {
        "Random Forest (Default)": rf_default,
        "Random Forest (Tuned)": rf_tuned,
        "SVM (Default)": svm_default,
        "SVM (Tuned)": svm_tuned,
        "LogReg (Default)": logreg_default,
        "LogReg (Tuned)": logreg_tuned,
    }

    for name, model in models.items():
        prediction = model.predict(X_scaled)[0]
        label = label_mapping.get(prediction, str(prediction))
        st.success(f"{name}: {label}")
