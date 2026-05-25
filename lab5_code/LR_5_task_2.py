import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

input_file = "data_imbalance.txt"

data = np.loadtxt(input_file, delimiter=",")

X = data[:, :-1]
y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5
)

# Без балансування
classifier = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=0
)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("WITHOUT BALANCE")
print(classification_report(y_test, y_pred))

# З балансуванням
classifier_balanced = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=0,
    class_weight="balanced"
)

classifier_balanced.fit(X_train, y_train)

y_pred_balanced = classifier_balanced.predict(X_test)

print("WITH BALANCE")
print(classification_report(y_test, y_pred_balanced))

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Imbalanced Classes")
plt.show()