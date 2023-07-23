import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

# Veri yükleme
data_url = "players_20.csv"  # veri dosyanızın yolunu buraya ekleyin
data = pd.read_csv(data_url)

# Türkçeleştirilmiş pozisyonlar
turkce_pozisyonlar = {
    'LM': 'Sol Kanat',
    'LB': 'Sol Bek',
    'CM': 'Orta Saha Orta',
    'RW': 'Sağ Açık',
    'ST': 'Forvet',
    'GK': 'Kaleci',
    'RB': 'Sağ Bek',
    'LW': 'Sol Açık',
    'CAM': 'Ofansif Orta Saha',
    'CDM': 'Defansif Orta Saha',
    'CB': 'Stoper',
    'RM': 'Sağ Kanat',
    'CF': 'Gizli Forvet',
    'LWB': 'Sol Kanat Arkası',
    'RWB': 'Sağ Kanat Arkası'
}

# Veriyi hazırlama
X = data[['age', 'player_positions', 'value_eur']]
y = data['overall']

# player_positions sütununu işleme
X['player_positions'] = X['player_positions'].apply(lambda x: ','.join(sorted(x.split(','))))

# player_positions için dummy değişkenleri oluşturma
all_positions = set([pos for positions in X['player_positions'] for pos in positions.split(',')])
for pos in all_positions:
    X[pos] = X['player_positions'].apply(lambda x: int(pos in x.split(',')))

X = X.drop(columns=['player_positions'])

# Eğitim ve test verilerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Makine öğrenmesi modelleri
models = {
    'XGBoost': xgb.XGBRegressor(),
    'LightGBM': lgb.LGBMRegressor(),
    'CatBoost': CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, loss_function='RMSE'),
    'KNN': KNeighborsRegressor(n_neighbors=5)
}

# Streamlit uygulamasını oluşturma
def main():
    st.title("Oyuncu Güç Tahmini")
    st.write("Bu uygulamada, oyuncunun yaşını, oynayabildiği bölgeleri ve market değerini kullanarak oyuncu gücü tahmin edilecektir.")
    st.write("Makine Öğrenmesi Modelleri: XGBoost, LightGBM, CatBoost, K-Nearest Neighbors (KNN)")

    # Girişleri alın
    selected_model = st.selectbox("Makine Öğrenmesi Modeli Seçin", list(models.keys()))
    age = st.slider("Oyuncu Yaşı", min_value=16, max_value=45, value=25, step=1)
    player_positions = st.multiselect("Oyuncunun Oynayabildiği Bölgeler", list(turkce_pozisyonlar.values()), default=list(turkce_pozisyonlar.values())[:2])
    value_eur = st.slider("Oyuncu Market Değeri (EUR)", min_value=0, max_value=200000000, value=5000000, step=10000)
    

    # player_positions verisini dizi olarak düzenleme
    input_data = pd.DataFrame([[age, value_eur]], columns=['age', 'value_eur'])
    for pos in all_positions:
        input_data[pos] = int(pos in player_positions)

    # Seçilen modeli oluşturma ve eğitme
    model = models[selected_model]
    model.fit(X_train, y_train)

    # R2 skoru ile model performansını değerlendirme
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    # Tahmin yapma
    prediction = model.predict(input_data)[0]

    st.subheader("Oyuncu Gücü Tahmini")
    st.write(f"Tahmin Edilen Oyuncu Gücü: {round(prediction, 2)}")

    # Model performansını gösterme
    st.subheader("Model Performansı (R2 Score)")
    st.write(f"R2 Skoru: {round(r2, 4)}")

if __name__ == "__main__":
    main()
