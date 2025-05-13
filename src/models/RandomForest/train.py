import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV


# Шаг 1: Загрузка данных
csv_file_path = "../../../data/processed/csv/eye_disease.csv"
data = pd.read_csv(csv_file_path, delimiter=";")
print("OK")

# Шаг 2: Подготовка данных
X = data.drop(columns=["ID", "is_Good"])
y = data["is_Good"]

# Шаг 3: Нормализация данных перед PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Шаг 4: Применение PCA
pca = PCA(n_components=9)  # Выбираем 9 компонент (можно изменить)
X_pca = pca.fit_transform(X_scaled)

# Визуализация объясненной дисперсии
plt.figure(figsize=(8, 5))
n_components = len(pca.explained_variance_ratio_)  # Определяем, сколько главных компонент
plt.plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_), marker="o", linestyle="--")
plt.xlabel("Количество компонент")
plt.ylabel("Накопленная объясненная дисперсия")
plt.title("График объясненной дисперсии PCA")
plt.show()

# Шаг 5: Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Шаг 6: Обучение модели Random Forest на PCA-преобразованных данных


param_dist = {
    'n_estimators': [200, 300, 400],
    'max_depth': [None, 15, 25],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', 0.6]
}

search = RandomizedSearchCV(
    estimator=RandomForestClassifier(),
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring='accuracy'
)

search.fit(X_train, y_train)
best_params = search.best_params_

rf_classifier = RandomForestClassifier(n_estimators=best_params["n_estimators"], min_samples_split=best_params["min_samples_split"], max_features=best_params["max_features"],max_depth=best_params["max_depth"])
rf_classifier.fit(X_train, y_train)

# Шаг 7: Прогнозирование и оценка модели
y_pred = rf_classifier.predict(X_test)
print("Точность модели:", accuracy_score(y_test, y_pred))
print("\nОтчет по классификации:\n", classification_report(y_test, y_pred))


# Шаг 8: Сохранение модели
import joblib

joblib.dump(rf_classifier, "random_forest_model.pkl")

print("Модель сохранена!")



