import pandas as pd
import xgboost as xgb
import joblib
import streamlit as st
import numpy as np

# Eğitilmiş XGBoost modelini yükleme
model = joblib.load("xgboost_model.pkl")

# Streamlit uygulamasını başlatma ve sayfa yapılandırmasını ayarlama
st.set_page_config(page_title="Futbol Oyuncusu Overall Tahmin Uygulaması")

# Kullanıcıdan giriş verilerini al
st.sidebar.header("Giriş Verileri")
value_eur = st.sidebar.number_input("Piyasa Değeri (EUR)", min_value=0)
age = st.sidebar.number_input("Yaş", min_value=0, max_value=100)

# Oynadığı pozisyonları açılır pencereden seçme
positions = st.sidebar.multiselect("Oynadığı Pozisyonlar", [
    'LB', 'RB', 'RM', 'CF', 'CDM', 'CAM', 'LM', 'GK', 'LWB', 'RWB', 'CM', 'RW', 'LW', 'CB', 'ST'
])

# Seçilen pozisyonları işaretleme
selected_positions = ['LB', 'RB', 'RM', 'CF', 'CDM', 'CAM', 'LM', 'GK', 'LWB', 'RWB', 'CM', 'RW', 'LW', 'CB', 'ST']
position_encoding = [1 if pos in positions else 0 for pos in selected_positions]

# Kullanıcının tahmin yapmasını tetikleme
if st.sidebar.button("Tahmin Yap"):
    # Kullanıcının girdiği verileri modele uygun bir veri çerçevesine dönüştürme
    input_data = pd.DataFrame({
        'value_eur': [value_eur],
        'age': [age]
    })

    # Seçilen pozisyonları sıfır yaparak işaretleme
    input_data[selected_positions] = position_encoding

    # Modelin tahmin yapabilmesi için giriş verilerini NumPy dizilerine dönüştürme
    input_data = input_data.values

    # Tahmin yapma
    overall_prediction = model.predict(input_data)

    # Tahmin sonucunu ekranda gösterme
    st.subheader("Tahmini Overall Değer:")
    st.write(f"{overall_prediction[0]:.2f}")
