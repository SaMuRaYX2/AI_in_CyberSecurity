import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split

housing_data = datasets.fetch_california_housing()

X = housing_data.data
y = housing_data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7
)

regressor = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4),
    n_estimators=400,
    random_state=7
)

regressor.fit(X_train, y_train)

feature_importances = regressor.feature_importances_

feature_importances = 100.0 * (
    feature_importances / max(feature_importances)
)

feature_names = np.array(housing_data.feature_names)

index_sorted = np.flipud(np.argsort(feature_importances))

pos = np.arange(index_sorted.shape[0]) + 0.5

plt.figure(figsize=(10, 6))

plt.bar(pos, feature_importances[index_sorted], align="center")

plt.xticks(pos, feature_names[index_sorted], rotation=45)

plt.ylabel("Relative Importance")

plt.title("Feature Importance using AdaBoost")

plt.tight_layout()

plt.show()