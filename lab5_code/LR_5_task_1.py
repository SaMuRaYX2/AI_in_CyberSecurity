import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Завантаження даних
input_file = "data_random_forests.txt"

data = np.loadtxt(input_file, delimiter=",")

X = data[:, :-1]
y = data[:, -1]

# Розбиття даних
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5
)

# Вибір класифікатора
classifier_type = "rf"   # rf або erf

params = {
    "n_estimators": 100,
    "max_depth": 4,
    "random_state": 0
}

if classifier_type == "rf":
    classifier = RandomForestClassifier(**params)
else:
    classifier = ExtraTreesClassifier(**params)

# Навчання
classifier.fit(X_train, y_train)

# Прогноз
y_pred = classifier.predict(X_test)

# Оцінка
print(classification_report(y_test, y_pred))

# Графік
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Random Forest / Extra Trees")
plt.show()