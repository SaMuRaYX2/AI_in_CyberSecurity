import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()

X = diabetes.data
y = diabetes.target

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.5, random_state=0
)

regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)

ypred = regr.predict(Xtest)

print("Коефіцієнти регресії:")
print(regr.coef_)

print("\nВільний член intercept:")
print(regr.intercept_)

print("\nR2 score:", round(r2_score(ytest, ypred), 4))
print("MAE:", round(mean_absolute_error(ytest, ypred), 4))
print("MSE:", round(mean_squared_error(ytest, ypred), 4))

fig, ax = plt.subplots()
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=4)
ax.set_xlabel("Виміряно")
ax.set_ylabel("Передбачено")
ax.set_title("Diabetes: виміряні та передбачені значення")
plt.grid(True)
plt.show()