import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

input_file = "data_random_forests.txt"

data = np.loadtxt(input_file, delimiter=",")

X = data[:, :-1]
y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5
)

parameter_grid = {
    "n_estimators": [25, 50, 100],
    "max_depth": [2, 4, 7]
}

classifier = GridSearchCV(
    ExtraTreesClassifier(random_state=0),
    parameter_grid,
    cv=5
)

classifier.fit(X_train, y_train)

print("Best parameters:")
print(classifier.best_params_)

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))