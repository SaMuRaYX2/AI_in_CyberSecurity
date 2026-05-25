import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"

data = pd.read_csv(url)

print("Перші 5 рядків:")
print(data.head())

print("\nІнформація про датасет:")
print(data.info())

# Видаляємо рядки без ціни
data = data.dropna(subset=["price"])

# Створюємо категорії ціни
data["price_category"] = pd.cut(
    data["price"],
    bins=3,
    labels=["low", "medium", "high"]
)

# Вибираємо ознаки
features = ["origin", "destination", "train_type", "train_class", "fare"]

# Залишаємо тільки потрібні стовпці
data = data[features + ["price_category"]]

# Видаляємо пропущені значення
data = data.dropna()

# Кодування текстових ознак
label_encoders = {}

for column in features:
    encoder = LabelEncoder()
    data[column] = encoder.fit_transform(data[column])
    label_encoders[column] = encoder

# Кодування цільової змінної
target_encoder = LabelEncoder()
data["price_category"] = target_encoder.fit_transform(data["price_category"])

X = data[features]
y = data["price_category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))