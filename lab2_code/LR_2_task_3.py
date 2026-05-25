from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
dataset = read_csv(url, names=names)

print("Shape:")
print(dataset.shape)

print("\nFirst 20 rows:")
print(dataset.head(20))

print("\nDescription:")
print(dataset.describe())

print("\nClass distribution:")
print(dataset.groupby("class").size())

dataset.plot(kind="box", subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.suptitle("Boxplot of Iris attributes")
pyplot.savefig("iris_boxplot.png")
pyplot.show()

dataset.hist()
pyplot.suptitle("Histogram of Iris attributes")
pyplot.savefig("iris_histogram.png")
pyplot.show()

scatter_matrix(dataset)
pyplot.suptitle("Scatter matrix of Iris dataset")
pyplot.savefig("iris_scatter_matrix.png")
pyplot.show()

array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]

X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y, test_size=0.20, random_state=1
)

models = []
models.append(("LR", LogisticRegression(solver="liblinear", multi_class="ovr")))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
models.append(("NB", GaussianNB()))
models.append(("SVM", SVC(gamma="auto")))

results = []
names = []

print("\nModel comparison:")
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

pyplot.boxplot(results, labels=names)
pyplot.title("Algorithm Comparison")
pyplot.savefig("iris_algorithm_comparison.png")
pyplot.show()

model = SVC(gamma="auto")
model.fit(X_train, Y_train)

predictions = model.predict(X_validation)

print("\nAccuracy on validation:")
print(accuracy_score(Y_validation, predictions))

print("\nConfusion matrix:")
print(confusion_matrix(Y_validation, predictions))

print("\nClassification report:")
print(classification_report(Y_validation, predictions))

X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = model.predict(X_new)

print("\nNew flower:")
print("X_new shape:", X_new.shape)
print("Prediction:", prediction[0])