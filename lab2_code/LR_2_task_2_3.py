from LR_2_task_1 import X_data, y
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y, test_size=0.2, random_state=5
)

classifier = SVC(kernel="sigmoid")
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Sigmoid SVM")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Precision:", round(precision_score(y_test, y_pred, average="weighted") * 100, 2), "%")
print("Recall:", round(recall_score(y_test, y_pred, average="weighted") * 100, 2), "%")
print("F1:", round(f1_score(y_test, y_pred, average="weighted") * 100, 2), "%")