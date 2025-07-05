import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # Diperlukan jika melatih ulang di sini
from sklearn.neighbors import KNeighborsClassifier # Diperlukan jika melatih ulang di sini
from sklearn.cluster import KMeans # Diperlukan jika melatih ulang di sini
import app_klasifikasi # Mengubah nama 'app' menjadi 'app_klasifikasi' untuk menghindari konflik
import app_clustering_gerai # Mengubah nama 'app_clustering' menjadi 'app_clustering_gerai'

# --- Fungsi untuk Memuat/Melatih Model dan Data ---
@st.cache_resource # Cache resource agar tidak memuat ulang/melatih ulang setiap kali aplikasi di-rerun
def load_and_train_models():
    models_and_data = {}

    # --- Untuk Klasifikasi Diabetes ---
    try:
        df_diabetes = pd.read_csv("diabetes.csv")
        st.sidebar.success("Dataset Diabetes berhasil dimuat.")

        # Preprocessing & Split Data Diabetes
        X_diabetes = df_diabetes.drop("Outcome", axis=1)
        y_diabetes = df_diabetes["Outcome"]

        scaler_diabetes = StandardScaler()
        X_scaled_diabetes = scaler_diabetes.fit_transform(X_diabetes)

        X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes = train_test_split(
            X_scaled_diabetes, y_diabetes, test_size=0.2, random_state=42
        )

        # Latih Model KNN Diabetes
        k_diabetes = 5 # Bisa disesuaikan
        model_diabetes = KNeighborsClassifier(n_neighbors=k_diabetes)
        model_diabetes.fit(X_train_diabetes, y_train_diabetes)
        
        models_and_data['model_diabetes'] = model_diabetes
        models_and_data['scaler_diabetes'] = scaler_diabetes
        models_and_data['X_test_diabetes'] = X_test_diabetes
        models_and_data['y_test_diabetes'] = y_test_diabetes
        models_and_data['diabetes_features'] = X_diabetes.columns.tolist() # Simpan nama fitur

    except FileNotFoundError:
        st.sidebar.error("❌ File 'diabetes.csv' tidak ditemukan. Pastikan ada di direktori yang sama.")
        models_and_data['model_diabetes'] = None
        models_and_data['scaler_diabetes'] = None
        models_and_data['X_test_diabetes'] = None
        models_and_data['y_test_diabetes'] = None
        models_and_data['diabetes_features'] = None
    except Exception as e:
        st.sidebar.error(f"Error memuat/melatih model diabetes: {e}")
        models_and_data['model_diabetes'] = None
        models_and_data['scaler_diabetes'] = None
        models_and_data['X_test_diabetes'] = None
        models_and_data['y_test_diabetes'] = None
        models_and_data['diabetes_features'] = None


    # --- Untuk Clustering Gerai ---
    try:
        df_gerai = pd.read_csv('lokasi_gerai_kopi_clean.csv') # Nama file sesuai app_clustering.py
        st.sidebar.success("Dataset Gerai berhasil dimuat.")

        # Tentukan fitur yang digunakan untuk clustering
        features_gerai = ['x', 'y', 'population_density', 'traffic_flow', 'competitor_count', 'is_commercial']
        
        if not all(f in df_gerai.columns for f in features_gerai):
            st.sidebar.error("❌ Kolom yang dibutuhkan untuk clustering gerai tidak lengkap dalam dataset.")
            models_and_data['model_gerai'] = None
            models_and_data['scaler_gerai'] = None
            models_and_data['df_gerai'] = None
            models_and_data['gerai_features'] = None
        else:
            df_clean_gerai = df_gerai[features_gerai].dropna()

            scaler_gerai = StandardScaler()
            X_scaled_gerai = scaler_gerai.fit_transform(df_clean_gerai)

            n_clusters_gerai = 5 # Bisa disesuaikan
            kmeans_gerai = KMeans(n_clusters=n_clusters_gerai, random_state=42, n_init=10) # n_init agar tidak DeprecatedWarning
            kmeans_gerai.fit(X_scaled_gerai)

            models_and_data['model_gerai'] = kmeans_gerai
            models_and_data['scaler_gerai'] = scaler_gerai
            models_and_data['df_gerai'] = df_clean_gerai # Gunakan df_clean_gerai yang sudah diproses
            models_and_data['gerai_features'] = features_gerai # Simpan nama fitur

    except FileNotFoundError:
        st.sidebar.error("❌ Dataset 'lokasi_gerai_kopi_clean.csv' tidak ditemukan.")
        models_and_data['model_gerai'] = None
        models_and_data['scaler_gerai'] = None
        models_and_data['df_gerai'] = None
        models_and_data['gerai_features'] = None
    except Exception as e:
        st.sidebar.error(f"Error memuat/melatih model gerai: {e}")
        models_and_data['model_gerai'] = None
        models_and_data['scaler_gerai'] = None
        models_and_data['df_gerai'] = None
        models_and_data['gerai_features'] = None

    return models_and_data

# Muat dan latih model saat aplikasi pertama kali dijalankan
st.set_page_config(page_title="Ujian Akhir Semester", layout="wide")
resources = load_and_train_models()

# Sidebar Navigasi
st.sidebar.title("📌 Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["🏁 Main Page", "📊 Klasifikasi", "📈 Clustering"])

# Halaman Utama
if page == "🏁 Main Page":
    st.title("Ujian Akhir Semester - Data Mining")
    st.header("📊 Aplikasi Analisis Data Menggunakan Streamlit")

    st.markdown("""
    **👤 Nama:** Mizan Ikbar  
    **🆔 NIM:** 22146003  
    **📁 Repository GitHub:** [Klik di sini](https://github.com/mizanikbar/streamlit-diabetess)
    """)

    st.markdown("---")
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=300)

# Halaman Klasifikasi
elif page == "📊 Klasifikasi":
    if resources['model_diabetes'] and resources['scaler_diabetes'] and resources['X_test_diabetes'] is not None:
        app_klasifikasi.show_classification_page(
            model=resources['model_diabetes'],
            scaler=resources['scaler_diabetes'],
            X_test=resources['X_test_diabetes'],
            y_test=resources['y_test_diabetes'],
            features_list=resources['diabetes_features']
        )
    else:
        st.error("Model Klasifikasi Diabetes atau data tidak dapat dimuat/dilatih.")

# Halaman Clustering
elif page == "📈 Clustering":
    if resources['model_gerai'] and resources['scaler_gerai'] and resources['df_gerai'] is not None:
        app_clustering_gerai.show_clustering_page(
            model=resources['model_gerai'],
            scaler=resources['scaler_gerai'],
            df=resources['df_gerai'],
            features_list=resources['gerai_features']
        )
    else:
        st.error("Model Clustering Gerai atau data tidak dapat dimuat/dilatih.")