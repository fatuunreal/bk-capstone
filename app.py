import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib


# === Load Trained Models ===
rf_default = joblib.load("rf_default.pkl")
rf_tuned = joblib.load("rf_tuned.pkl")
svm_default = joblib.load("svm_default.pkl")
svm_tuned = joblib.load("svm_tuned.pkl")
logreg_default = joblib.load("logreg_default.pkl")
logreg_tuned = joblib.load("logreg_tuned.pkl")
feature_order = joblib.load("feature_columns.pkl")

# === Load Scaler ===
scaler = joblib.load("scaler.pkl")

# === Label Mapping ===
label_mapping = {
    0: 'Insufficient_Weight',
    1: 'Normal_Weight',
    2: 'Overweight_Level_I',
    3: 'Overweight_Level_II',
    4: 'Obesity_Type_I',
    5: 'Obesity_Type_II',
    6: 'Obesity_Type_III'
}

# === Manual Model Accuracies ===
model_accuracies = {
    "Random Forest (Default)": 0.96,
    "Random Forest (Tuned)": 0.97,
    "SVM (Default)": 0.93,
    "SVM (Tuned)": 0.95,
    "LogReg (Default)": 0.92,
    "LogReg (Tuned)": 0.94
}

# === Streamlit App ===
st.title("Obesity Classification App")

st.write("Masukkan data pribadi dan kebiasaan Anda:")

# === Input Form ===
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 10, 100, 25)
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
    # === Encoding Inputs ===
    input_dict = {
        'Age': age,
        'Height': height,
        'Weight': weight,
        'FCVC': fcvc,
        'NCP': ncp,
        'CH2O': ch2o,
        'FAF': faf,
        'TUE': tue,
        'Gender_Male': 1 if gender == "Male" else 0,
        'family_history_with_overweight_yes': 1 if famhist == "yes" else 0,
        'FAVC_yes': 1 if favc == "yes" else 0,
        'SCC_yes': 1 if scc == "yes" else 0,
        'SMOKE_yes': 1 if smoke == "yes" else 0,
        'CALC_Always': int(calc == "Always"),
        'CALC_Frequently': int(calc == "Frequently"),
        'CALC_Sometimes': int(calc == "Sometimes"),
        'CAEC_Frequently': int(caec == "Frequently"),
        'CAEC_Sometimes': int(caec == "Sometimes"),
        'CAEC_Always': int(caec == "Always"),
        'MTRANS_Bike': int(mtrans == "Bike"),
        'MTRANS_Motorbike': int(mtrans == "Motorbike"),
        'MTRANS_Walking': int(mtrans == "Walking"),
        'MTRANS_Automobile': int(mtrans == "Automobile")
    }

    input_df = pd.DataFrame([input_dict])
    required_cols = scaler.feature_names_in_
    input_df = input_df.reindex(columns=feature_order, fill_value=0)
    X_scaled = scaler.transform(input_df)

    models = {
        "Random Forest (Default)": rf_default,
        "Random Forest (Tuned)": rf_tuned,
        "SVM (Default)": svm_default,
        "SVM (Tuned)": svm_tuned,
        "LogReg (Default)": logreg_default,
        "LogReg (Tuned)": logreg_tuned,
    }

    # === Tampilkan hasil klasifikasi sebagai tabel dan download ===
    results = {"Model": [], "Prediction": [], "Accuracy": []}
    for name, model in models.items():
        prediction = model.predict(X_scaled)[0]
        label = label_mapping[prediction] if prediction in label_mapping else str(prediction)
        acc = model_accuracies[name]
        results["Model"].append(name)
        results["Prediction"].append(label)
        results["Accuracy"].append(f"{acc*100:.2f}%")

    st.subheader("Hasil Prediksi Seluruh Model")
    result_df = pd.DataFrame(results)
    st.dataframe(result_df)

    # === Download Button ===
    result_text = "\n".join([f"{m}: {p} (Accuracy: {a})" for m, p, a in zip(results['Model'], results['Prediction'], results['Accuracy'])])
    st.download_button("Download Hasil Prediksi", result_text, file_name="hasil_prediksi.txt")

    # === Bar Chart ===
    st.subheader("Perbandingan Akurasi Model")
    st.bar_chart(pd.Series(model_accuracies).sort_values(ascending=False))

# === Keterangan Kategori ===
st.markdown("""
### Keterangan Kategori Obesitas:
- **Insufficient Weight**: Berat badan kurang
- **Normal Weight**: Berat badan ideal
- **Overweight Level I & II**: Kelebihan berat badan ringan hingga sedang
- **Obesity Type I/II/III**: Obesitas tingkat ringan sampai berat
""")
