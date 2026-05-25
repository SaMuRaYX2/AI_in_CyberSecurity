# Дані:
X = [13.33, 21, 63.75, 20.87, 40.42, 30.27]
Y = [10.48, 21.03, 23.02, 41.25, 27.16, 51.5]

# Наш код для реалізації завдання 2.1 Лінійна регресія

import numpy as np
import matplotlib.pyplot as plt

X = np.array([13.33, 21, 63.75, 20.87, 40.42, 30.27])
Y = np.array([10.48, 21.03, 23.02, 41.25, 27.16, 51.5])

# Знаходимо коефіцієнти прямої y = kx + b
k, b = np.polyfit(X, Y, 1)

print("Рівняння апроксимуючої прямої:")
print(f"y = {k:.4f}x + {b:.4f}")

Y_pred = k * X + b

SSE = np.sum((Y - Y_pred) ** 2)
MSE = np.mean((Y - Y_pred) ** 2)

print("Сума квадратів похибок SSE:", round(SSE, 4))
print("Середньоквадратична похибка MSE:", round(MSE, 4))

plt.scatter(X, Y, label="Експериментальні точки")

X_line = np.linspace(min(X), max(X), 100)
Y_line = k * X_line + b

plt.plot(X_line, Y_line, label="Апроксимуюча пряма")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Лінійна регресія методом найменших квадратів")
plt.legend()
plt.grid(True)
plt.show()