"""
Sistem Otomatisasi Preprocessing (MLOps Pipeline)
Proyek: Analisis Sentimen & Recipe Reviews
Pembuat: Mochammad Rafly Herdianto
"""

import pandas as pd
import numpy as np
import logging
import warnings
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Mematikan notifikasi peringatan bawaan Pandas agar hasil Terminal terlihat bersih dan profesional
warnings.filterwarnings('ignore')

# Menjadwalkan Format Logging layaknya Server-side Pipeline
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ %(levelname)s ] - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("===== MENGINISIASI PIPELINE PREPROCESSING OTOMATIS =====")

    # ---------------------------------------------------------
    # 1. PERSIAPAN JALUR DIREKTORI (PATHING UNTUK LOCAL & CLOUD)
    # ---------------------------------------------------------
    # Path Dinamis: Bisa bekerja di laptop (Windows) maupun di GitHub Actions (Ubuntu Linux)
    ROOT_DIR = Path(__file__).resolve().parent.parent
    RAW_FILE = ROOT_DIR / "recipereviews_raw" / "Recipe Reviews and User Feedback Dataset.csv"
    OUTPUT_DIR = ROOT_DIR / "preprocessing" / "recipereviews_preprocessing"

    # Memastikan lokasi folder sudah ada
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Target Penyimpanan Disetel: {OUTPUT_DIR}")

    # ---------------------------------------------------------
    # 2. PROSES PEMUATAN DATA (DATA INGESTION)
    # ---------------------------------------------------------
    try:
        logger.info("Mengambil Dataset Mentah dari direktori asalnya...")
        df = pd.read_csv(RAW_FILE)
    except FileNotFoundError:
        logger.error(f"Dataset Mentah Gagal Ditemukan di {RAW_FILE}. Harap pastikan kembali.")
        return

    # ------------------------------
    # 3. BASE CLEANING
    # ------------------------------
    logger.info("Menjalankan Cleaning...")
    
    # Penanggulangan Kolom Sampah tak ber-identitas
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Dekompresi Baris Ganda (Drop Duplicates)
    df = df.drop_duplicates()
    
    # Imputasi kolom kuantitatif kritis
    interaksi = ['reply_count', 'thumbs_up', 'thumbs_down', 'best_score', 'stars']
    for col in interaksi:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # ---------------------------------------------------------
    # 4. BINNING SENTIMEN & EKSTRAKSI WAKTU
    # ---------------------------------------------------------
    logger.info("Melakukan Feature Engineering...")
    
    # Penindasan bias rating kelas: Menghapus nilai Bintang "0", dan klasifikasikan sisanya
    if 'stars' in df.columns:
        df = df[df['stars'] != 0].copy()
        
        # 1-3 Diumpamakan Kelas 0 (Negatif), 4-5 Disimpan sebagai Kelas 1 (Positif)
        df['sentiment_class'] = df['stars'].apply(lambda x: 1 if x >= 4 else 0)

    # Rekonstruksi Waktu Berbasis Sistem Operasi (Unix) ke Sistem Almanak (Kalenderisasi)
    if 'created_at' in df.columns:
        waktu_baku = pd.to_datetime(df['created_at'], unit='s')
        df['review_year'] = waktu_baku.dt.year
        df['review_month'] = waktu_baku.dt.month

    # ---------------------------------------------------------
    # 5. PEMBUANGAN FITUR IDENTIFIKASI (DROP NOISE/ID)
    # ---------------------------------------------------------
    kolom_sampah_fitur = [
        'user_reputation', 'best_score', 'stars', 'waktu_baku', 
        'created_at', 'comment_id', 'user_id', 
        'recipe_code', 'user_name', 'recipe_name', 'recipe_number'
    ]
    df = df.drop(columns=[col for col in kolom_sampah_fitur if col in df.columns])

    # ---------------------------------------------------------
    # 6. PROSES ANTI-KEBOCORAN DATA (SPLIT)
    # ---------------------------------------------------------
    logger.info("Mengisolasi Batas Data (Aman Dari Leakage Data) (Train-Test Split)...")
    target = 'sentiment_class'
    
    if target not in df.columns:
         logger.error("Wah, Target Sentimen Klasifikasi hilang! Cek algoritma binning di atas.")
         return

    X = df.drop(columns=[target])
    y = df[target]

    # Memecah secara Stratified agar pecahan Minoritas tetap ada di kumpulan Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ---------------------------------
    # 7. STANDARISASI DATA KUANTITATIF
    # ---------------------------------
    logger.info("Proyeksi Normalisasi Skala Menggunakan RobustScaler...")
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    if num_cols:
        scaler = RobustScaler()
        # Hitung Mean&Std di Data Uji SAJA, Transform pada Semuanya!
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols]  = scaler.transform(X_test[num_cols])
        
        scaler_file = OUTPUT_DIR / 'robust_scaler_production.pkl'
        if scaler_file.exists():
            logger.warning(f"File lama terdeteksi, overwrite file: {scaler_file.name}")
        joblib.dump(scaler, scaler_file)

    # ---------------------------------------------------------
    # 8. KONVERSI BAHASA ALAMI (NLP ENGAGEMENT)
    # ---------------------------------------------------------
    logger.info("Dekomposisi Baris Subjektif Resep menjadi Numerik via TF-IDF...")
    if 'text' in X_train.columns:
        # Membenahi kebolongan teks sisa ke string kosong ""
        X_train['text'] = X_train['text'].fillna("")
        X_test['text']  = X_test['text'].fillna("")
        
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english', min_df=5, max_df=0.9)
        
        X_train_text = tfidf.fit_transform(X_train['text']).toarray()
        X_test_text  = tfidf.transform(X_test['text']).toarray()
        
        tfidf_file = OUTPUT_DIR / 'tfidf_model_production.pkl'
        if tfidf_file.exists():
            logger.warning(f"File lama terdeteksi, overwrite file: {tfidf_file.name}")
        joblib.dump(tfidf, tfidf_file)
        
        fitur_nlp = [f"kata_{w}" for w in tfidf.get_feature_names_out()]
        df_train_nlp = pd.DataFrame(X_train_text, columns=fitur_nlp, index=X_train.index)
        df_test_nlp  = pd.DataFrame(X_test_text, columns=fitur_nlp, index=X_test.index)
        
        # Penyatuan Final Dimensi
        X_train = pd.concat([X_train.drop(columns=['text']), df_train_nlp], axis=1)
        X_test  = pd.concat([X_test.drop(columns=['text']), df_test_nlp], axis=1)

    # ---------------------------------------------------------
    # 9. OUTPUT & AKHIRI PROGRAM
    # ---------------------------------------------------------
    logger.info("Menyimpan Matriks Data Bersih ke Disk...")
    
    def save_csv_overwrite(df_obj, filename):
        filepath = OUTPUT_DIR / filename
        if filepath.exists():
            logger.warning(f"File lama terdeteksi, overwrite file CSV: {filename}")
        df_obj.to_csv(filepath, index=False, mode='w') # mode='w' mereset ulang seluruh isi file lama

    save_csv_overwrite(X_train, 'X_train_ready.csv')
    save_csv_overwrite(X_test, 'X_test_ready.csv')
    save_csv_overwrite(y_train, 'y_train_ready.csv')
    save_csv_overwrite(y_test, 'y_test_ready.csv')

    logger.info(f"Dimensi Akhir -> Latih (Train): {X_train.shape}, Uji (Test): {X_test.shape}")
    logger.info("===== PIPELINE SELESAI ! =====")

# Memastikan Script Hanya Berjalan Jika Tidak Diimpor Aplikasi Lain
if __name__ == "__main__":
    main()