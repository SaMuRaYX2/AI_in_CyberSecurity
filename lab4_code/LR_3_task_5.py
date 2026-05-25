import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.6 * X ** 2 + X + 2 + np.random.randn(m, 1)

lin_reg = LinearRegression()
lin_reg.fit(X, y)

X_line = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_lin_pred = lin_reg.predict(X_line)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

X_line_poly = poly_features.transform(X_line)
y_poly_pred = poly_reg.predict(X_line_poly)

y_poly_train_pred = poly_reg.predict(X_poly)

print("Початкове X[0]:")
print(X[0])

print("\nX_poly[0]:")
print(X_poly[0])

print("\nЛінійна модель:")
print("intercept =", lin_reg.intercept_)
print("coef =", lin_reg.coef_)

print("\nПоліноміальна модель:")
print("intercept =", poly_reg.intercept_)
print("coef =", poly_reg.coef_)

print("\nОцінка поліноміальної регресії:")
print("MAE =", round(mean_absolute_error(y, y_poly_train_pred), 4))
print("MSE =", round(mean_squared_error(y, y_poly_train_pred), 4))
print("R2 =", round(r2_score(y, y_poly_train_pred), 4))

plt.scatter(X, y, label="Дані")
plt.plot(X_line, y_lin_pred, label="Лінійна регресія")
plt.plot(X_line, y_poly_pred, label="Поліноміальна регресія degree=2")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Лінійна та поліноміальна регресія")
plt.legend()
plt.grid(True)
plt.show()