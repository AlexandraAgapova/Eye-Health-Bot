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

cnt_healthy = 0
cnt_disease = 0

for i in range(len(data)):
    if prediction[i] == 0:
        cnt_disease+=1
    else:
        cnt_healthy+=1

if cnt_disease > cnt_healthy:
    print("Your face is healthy!)")
else:
    print("Your face is bad(")
          