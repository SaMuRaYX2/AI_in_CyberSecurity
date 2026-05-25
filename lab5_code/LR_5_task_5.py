import numpy as np

from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

input_file = "traffic_data.txt"

data = []

with open(input_file, "r") as f:
    for line in f.readlines():
        items = line[:-1].split(",")
        data.append(items)

data = np.array(data)

label_encoder = []

X_encoded = np.empty(data.shape)

for i, item in enumerate(data[0]):
    if item.isdigit():
        X_encoded[:, i] = data[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(data[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5
)

regressor = ExtraTreesRegressor(
    n_estimators=100,
    max_depth=4,
    random_state=0
)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print("Mean absolute error:")
print(round(mean_absolute_error(y_test, y_pred), 2))