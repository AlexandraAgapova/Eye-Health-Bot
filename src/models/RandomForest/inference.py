import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Загрузка модели
rf_model = joblib.load("random_forest_model.pkl")

# Использование модели для предсказаний
new_data = pd.read_csv("test.csv", delimiter=";")  # Данные в таком же формате, как при обучении
data = new_data.drop(columns=["ID", "is_Good"])
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)


prediction = rf_model.predict(data_scaled)  # Без PCA


print("Предсказание:", prediction)