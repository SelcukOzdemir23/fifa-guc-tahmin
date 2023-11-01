import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error ,accuracy_score
import joblib

# Verileri yükleyin (örnek olarak "data.csv" dosyasından alınmış varsayalım)
data = pd.read_csv("data/players_20.csv")


# Veriyi özellikler ve hedef değişken olarak ayırma
X = data[['value_eur', 'age', 'player_positions']]
y = data['overall']

# Kategorik sütunu One-Hot Encoding ile dönüştürme
dummies = pd.get_dummies(X["player_positions"])

X.drop(["player_positions"],axis=1,inplace=True)
X = pd.concat([X,dummies],axis=1)

# Veriyi eğitim ve test kümelerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost modelini eğitme
model = xgb.XGBRegressor()
model.fit(X_train, y_train)


# Test kümesi üzerinde modelin performansını değerlendirme
y_pred_test = model.predict(X_test)
test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
print("Test Kümesi RMSE:", test_rmse)

# Modeli kaydetme
joblib.dump(model, "xgboost_model.pkl")
