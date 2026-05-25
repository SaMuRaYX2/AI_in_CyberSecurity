x = [0.1, 0.3, 0.4, 0.6, 0.7]
y = [3.2, 3, 1, 1.8, 1.9]

import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
y = np.array([3.2, 3.0, 1.0, 1.8, 1.9])

# Поліном 4-го степеня
coefficients = np.polyfit(x, y, 4)

print("Коефіцієнти інтерполяційного полінома:")
print(coefficients)

poly = np.poly1d(coefficients)

print("Інтерполяційний поліном:")
print(poly)

y_02 = poly(0.2)
y_05 = poly(0.5)

print("Значення функції в точці x = 0.2:", round(y_02, 4))
print("Значення функції в точці x = 0.5:", round(y_05, 4))

x_new = np.linspace(min(x), max(x), 200)
y_new = poly(x_new)

plt.scatter(x, y, label="Задані точки")
plt.plot(x_new, y_new, label="Інтерполяційний поліном 4-го степеня")
plt.scatter([0.2, 0.5], [y_02, y_05], label="Проміжні точки")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Інтерполяція поліномом 4-го степеня")
plt.legend()
plt.grid(True)
plt.show()