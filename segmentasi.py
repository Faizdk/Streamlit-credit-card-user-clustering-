import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Mapping nama model yang ditampilkan ke nama file
MODEL_OPTIONS = {
    "K-Means (2 clusters)": "kmeans_k2.pkl",
    "K-Means (3 clusters)": "kmeans_k3.pkl",
    "K-Means (4 clusters)": "kmeans_k4.pkl"
}

# Load semua model dan scaler sekali di awal 
@st.cache_resource
def load_models_and_scaler():
    models = {}
    for display_name, filename in MODEL_OPTIONS.items():
        try:
            models[display_name] = joblib.load(filename)
        except Exception as e:
            st.error(f"Gagal memuat model {filename}: {e}")
    try:
        scaler = joblib.load("scaler.pkl")
    except Exception as e:
        st.error(f"Gagal memuat scaler: {e}")
        scaler = None
    return models, scaler


# ────────────────────────────────────────────────
# Aplikasi utama
# ────────────────────────────────────────────────

st.title("Credit Card Customer Segmentation")

st.write("Masukkan data pelanggan untuk prediksi segmentasi")

# Load model & scaler
models, scaler = load_models_and_scaler()

if not models:
    st.stop()

if scaler is None:
    st.stop()

# Pilihan model
selected_model_name = st.selectbox(
    "Pilih jumlah cluster / model yang digunakan",
    options=list(MODEL_OPTIONS.keys()),
    index=1  # default ke K=3
)

# Penjelasan yang berubah secara dinamis sesuai pilihan model
if "2 clusters" in selected_model_name:
    st.info("""
    **K = 2** — Pembagian luas menjadi dua kelompok besar:  
    Memiliki skor akurasi tertinggi dibandingkan model lainnya, cocok untuk pembedaan sederhana, misalnya pelanggan bernilai tinggi vs. bernilai rendah.
    """)
elif "3 clusters" in selected_model_name:
    st.info("""
    **K = 3** — Menghasilkan pembagian yang lebih bernuansa:  
    Biasanya memisahkan kelompok menengah dari salah satu kelompok besar sebelumnya.
    """)
elif "4 clusters" in selected_model_name:
    st.info("""
    **K = 4** — Segmentasi menjadi lebih granular dan spesifik:  
    Dapat mengungkap sub-segmen perilaku pelanggan yang lebih detail, namun tingkat skor akurasi lebih rendah.
    """)
else:
    st.caption("Pilih model di atas untuk melihat penjelasan.")
# Ambil model yang dipilih
kmeans_model = models[selected_model_name]


# ── Input data ────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    annual_income = st.number_input("Annual Income", min_value=0, value=50000, step=1000)
    total_spend = st.number_input("Total Spend Last Year", min_value=0, value=10000, step=500)

with col2:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700, step=10)
    clv = st.number_input("CLV (Customer Lifetime Value)", min_value=0, value=2500, step=100)

# ── Prediksi ──────────────────────────────────────────
if st.button("Predict Segment", type="primary"):

    # Buat DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Annual_Income': [annual_income],
        'Total_Spend_Last_Year': [total_spend],
        'Credit_Score': [credit_score],
        'CLV': [clv]
    })

    # Scaling
    try:
        input_scaled = scaler.transform(input_data)
    except Exception as e:
        st.error(f"Error saat scaling data: {e}")
        st.stop()

    # Prediksi
    try:
        segment = kmeans_model.predict(input_scaled)[0]
    except Exception as e:
        st.error(f"Error saat prediksi: {e}")
        st.stop()

    # Tampilkan hasil
    st.success(f"**Hasil Prediksi:** {selected_model_name}")
    st.subheader(f"**Segmen Cluster {segment}**")

# ── Penjelasan hanya untuk segmen yang diprediksi ──

    if "2 clusters" in selected_model_name:
        if segment == 0:
            st.markdown("""
            **Cluster 0** → Customer dengan income **tinggi**, spending **tinggi**, nilai CLV **tinggi**, skor kredit **sehat**, peluang defaulted ~34.96%.
            """)
        elif segment == 1:
            st.markdown("""
            **Cluster 1** → Customer dengan income **rendah**, spending **rendah**, nilai CLV **rendah**, skor kredit **sehat**, peluang defaulted ~34.22%.
            """)

    elif "3 clusters" in selected_model_name:
        if segment == 0:
            st.markdown("""
            **Cluster 0** → Customer **muda** dengan income **moderat**, spending **rendah**, nilai CLV **rendah**, skor kredit **sehat**, peluang defaulted ~33.66% (terendah).
            """)
        elif segment == 1:
            st.markdown("""
            **Cluster 1** → Customer dengan income **tinggi**, spending **tinggi**, nilai CLV **tinggi**, skor kredit **sehat**, peluang defaulted ~34.75%.
            """)
        elif segment == 2:
            st.markdown("""
            **Cluster 2** → Customer **tua** dengan income **moderat**, spending **rendah**, nilai CLV **rendah**, skor kredit **sehat**, peluang defaulted ~35.15% (tertinggi).
            """)

    elif "4 clusters" in selected_model_name:
        if segment == 0:
            st.markdown("""
            **Cluster 0** → Customer dengan income **sangat rendah**, spending **sangat rendah**, nilai CLV **sangat rendah**, skor kredit **agak rendah**, peluang defaulted ~34.79%.
            """)
        elif segment == 1:
            st.markdown("""
            **Cluster 1** → Customer dengan income **sangat tinggi**, spending **sangat tinggi**, nilai CLV **sangat tinggi**, skor kredit **sehat**, peluang defaulted ~35.63% (tertinggi).
            """)
        elif segment == 2:
            st.markdown("""
            **Cluster 2** → Customer **tua** dengan income **tinggi**, spending **menengah**, nilai CLV **menengah**, skor kredit **tertinggi**, peluang defaulted ~34.30%.
            """)
        elif segment == 3:
            st.markdown("""
            **Cluster 3** → Customer **muda** dengan income **moderat**, spending **menengah**, nilai CLV **rendah**, skor kredit **sehat**, peluang defaulted ~33.76% (terendah).
            """)

    # Garis pemisah opsional
    st.markdown("---")
    st.caption("Deskripsi berdasarkan karakteristik relatif kluster pada data pelatihan.")
