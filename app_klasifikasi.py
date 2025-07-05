import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def show_classification_page(model, scaler, X_test, y_test, features_list):
    st.title("📊 Klasifikasi Diabetes - KNN")
    st.write("Aplikasi ini memprediksi apakah seseorang terkena diabetes menggunakan model **K-Nearest Neighbors (KNN)**.")

    st.subheader("📈 Metrik Klasifikasi (Data Test)")
    if model and scaler is not None and X_test is not None and y_test is not None:
        try:
            y_pred = model.predict(X_test)
            st.text(classification_report(y_test, y_pred))
        except Exception as e:
            st.error(f"Error saat menghitung classification report: {e}")
            st.info("Pastikan data test dan model kompatibel.")
    else:
        st.warning("Model atau data test tidak lengkap untuk menampilkan metrik.")


    st.subheader("🧮 Confusion Matrix")
    if model and scaler is not None and X_test is not None and y_test is not None:
        try:
            y_pred = model.predict(X_test)
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Tidak Diabetes', 'Diabetes'], yticklabels=['Tidak Diabetes', 'Diabetes'])
            plt.xlabel("Prediksi")
            plt.ylabel("Aktual")
            plt.title("Confusion Matrix")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error saat menampilkan confusion matrix: {e}")
    else:
        st.warning("Model atau data test tidak lengkap untuk menampilkan confusion matrix.")


    # Input Data Baru
    st.subheader("📝 Input Data Baru untuk Prediksi")
    
    # Mapping nama fitur dari file diabetes.csv ke label yang lebih user-friendly
    feature_labels = {
        'Pregnancies': 'Jumlah Kehamilan',
        'Glucose': 'Glukosa (mg/dL)',
        'BloodPressure': 'Tekanan Darah (mmHg)',
        'SkinThickness': 'Ketebalan Kulit (mm)',
        'Insulin': 'Insulin (muU/ml)',
        'BMI': 'BMI',
        'DiabetesPedigreeFunction': 'Fungsi Silsilah Diabetes',
        'Age': 'Usia'
    }

    # Asumsi urutan input sesuai dengan fitur model
    input_values = {}
    col1, col2 = st.columns(2)
    
    # Pastikan urutan dan jumlah input sesuai dengan urutan fitur dalam model
    # Jika features_list tidak ada (misal karena error load), pakai fallback default
    if features_list:
        for i, feature in enumerate(features_list):
            label = feature_labels.get(feature, feature) # Ambil label user-friendly atau gunakan nama fitur
            if i % 2 == 0:
                with col1:
                    if feature == 'Pregnancies':
                        input_values[feature] = st.number_input(label, min_value=0, value=1)
                    elif feature == 'Age':
                        input_values[feature] = st.number_input(label, min_value=21, value=33)
                    elif feature in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']:
                        input_values[feature] = st.number_input(label, min_value=0.0, value=0.0) # Sesuaikan nilai default
            else:
                with col2:
                    if feature == 'Pregnancies':
                        input_values[feature] = st.number_input(label, min_value=0, value=1)
                    elif feature == 'Age':
                        input_values[feature] = st.number_input(label, min_value=21, value=33)
                    elif feature in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']:
                        input_values[feature] = st.number_input(label, min_value=0.0, value=0.0) # Sesuaikan nilai default
    else:
        st.warning("Daftar fitur tidak tersedia. Menggunakan input default. Pastikan fitur sesuai.")
        # Fallback jika features_list kosong/None
        input_values['Pregnancies'] = st.number_input("Jumlah Kehamilan (Pregnancies)", min_value=0)
        input_values['Glucose'] = st.number_input("Glukosa", min_value=0.0)
        input_values['BloodPressure'] = st.number_input("Tekanan Darah (Blood Pressure)", min_value=0.0)
        input_values['SkinThickness'] = st.number_input("Ketebalan Kulit (Skin Thickness)", min_value=0.0)
        input_values['Insulin'] = st.number_input("Insulin", min_value=0.0)
        input_values['BMI'] = st.number_input("BMI", min_value=0.0)
        input_values['DiabetesPedigreeFunction'] = st.number_input("Diabetes Pedigree Function", min_value=0.0)
        input_values['Age'] = st.number_input("Usia", min_value=0)
        features_list = list(input_values.keys()) # Update features_list for prediction


    input_data_df = pd.DataFrame([input_values]) # Buat DataFrame dari input values

    if st.button("🔍 Prediksi"):
        if model is None or scaler is None:
            st.error("Model atau Scaler belum dimuat. Tidak dapat melakukan prediksi.")
            return

        # Pastikan input_data_df memiliki kolom yang sama dengan fitur yang digunakan model
        # Reorder kolom jika perlu
        input_data_ordered = input_data_df[features_list]

        try:
            input_scaled = scaler.transform(input_data_ordered)
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)

            st.subheader("✅ Hasil Prediksi")
            if prediction[0] == 1:
                st.error(f"**Pasien kemungkinan menderita Diabetes.** (Probabilitas: {prediction_proba[0][1]*100:.2f}%)")
            else:
                st.success(f"**Pasien tidak menderita Diabetes.** (Probabilitas: {prediction_proba[0][0]*100:.2f}%)")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
            st.info("Pastikan format input sesuai dengan yang diharapkan model.")