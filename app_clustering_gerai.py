import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def show_clustering_page(model, scaler, df, features_list):
    st.title("📍 Clustering Lokasi Gerai Kopi - KMeans")
    st.write("Analisis lokasi gerai kopi menggunakan algoritma **K-Means Clustering**.")

    st.write("📌 Kolom dalam dataset:", df.columns.tolist())

    st.header("Visualisasi Clustering Berdasarkan Koordinat")
    if model and scaler is not None and df is not None:
        try:
            # Gunakan fitur yang digunakan untuk melatih model untuk prediksi klaster pada DF
            df_for_prediction = df[features_list]
            df_scaled_for_prediction = scaler.transform(df_for_prediction)
            
            # Tambahkan kolom cluster ke DataFrame untuk visualisasi
            df_display = df.copy() # Buat salinan agar tidak mengubah DataFrame asli
            df_display['cluster'] = model.predict(df_scaled_for_prediction)

            fig, ax = plt.subplots(figsize=(10, 6))
            # Asumsi 'x' dan 'y' adalah fitur untuk visualisasi
            if 'x' in df_display.columns and 'y' in df_display.columns:
                sns.scatterplot(data=df_display, x='x', y='y', hue='cluster', palette='tab10', s=60, ax=ax)
                plt.title(f"Hasil Clustering KMeans (K = {model.n_clusters})")
                plt.xlabel("X Coordinate")
                plt.ylabel("Y Coordinate")
                plt.grid(True)
                st.pyplot(fig)
            else:
                st.warning("Kolom 'x' dan 'y' tidak ditemukan untuk visualisasi koordinat.")
                st.write("Data dengan Klaster:")
                st.dataframe(df_display[['cluster'] + features_list]) # Tampilkan data dengan klaster

        except Exception as e:
            st.error(f"Terjadi kesalahan saat visualisasi clustering: {e}")
            st.info("Pastikan kolom 'x' dan 'y' ada di dataset gerai Anda dan model telah dilatih dengan benar.")
    else:
        st.warning("Model, scaler, atau data gerai tidak lengkap untuk visualisasi.")

    # Input lokasi baru
    st.subheader("📝 Input Lokasi Baru untuk Prediksi Cluster")
    
    # Asumsi fitur yang digunakan adalah: 'x', 'y', 'population_density', 'traffic_flow', 'competitor_count', 'is_commercial'
    # Pastikan urutan dan nama input sesuai dengan features_list yang digunakan model
    input_values = {}
    col1, col2 = st.columns(2)

    # Pastikan features_list ada dan digunakan untuk input
    if features_list:
        for i, feature in enumerate(features_list):
            if i % 2 == 0:
                with col1:
                    if feature == 'is_commercial':
                        input_values[feature] = st.selectbox(f"Apakah {feature_labels.get(feature, feature)}?", [0, 1])
                    else:
                        input_values[feature] = st.number_input(f"{feature_labels.get(feature, feature)}", value=0.0)
            else:
                with col2:
                    if feature == 'is_commercial':
                        input_values[feature] = st.selectbox(f"Apakah {feature_labels.get(feature, feature)}?", [0, 1])
                    else:
                        input_values[feature] = st.number_input(f"{feature_labels.get(feature, feature)}", value=0.0)
    else:
        st.warning("Daftar fitur clustering tidak tersedia. Menggunakan input default. Pastikan fitur sesuai.")
        # Fallback jika features_list kosong/None
        input_values['x'] = st.number_input("X Coordinate", value=0.0)
        input_values['y'] = st.number_input("Y Coordinate", value=0.0)
        input_values['population_density'] = st.number_input("Population Density", value=0.0)
        input_values['traffic_flow'] = st.number_input("Traffic Flow", value=0.0)
        input_values['competitor_count'] = st.number_input("Jumlah Kompetitor di Sekitar", value=0)
        input_values['is_commercial'] = st.selectbox("Apakah Area Komersial?", [0, 1])
        features_list = list(input_values.keys()) # Update features_list for prediction

    # Membuat DataFrame input dengan urutan kolom yang benar
    input_data = pd.DataFrame([input_values], columns=features_list)

    if st.button("🔍 Prediksi Cluster"):
        if model is None or scaler is None:
            st.error("Model atau Scaler belum dimuat. Tidak dapat melakukan prediksi.")
            return

        try:
            input_scaled = scaler.transform(input_data)
            predicted_cluster = model.predict(input_scaled)
            st.success(f"✅ Lokasi baru ini termasuk dalam **Klaster {predicted_cluster[0]}**")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi cluster: {e}")
            st.info("Pastikan input Anda sesuai dengan fitur dan urutan yang diharapkan model.")