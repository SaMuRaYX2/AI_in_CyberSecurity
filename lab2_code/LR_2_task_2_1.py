from LR_2_task_1 import X_data, y

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Беремо менше даних, щоб програма не зависала
X_small = X_data[:5000]
y_small = y[:5000]

X_train, X_test, y_train, y_test = train_test_split(
    X_small, y_small, test_size=0.2, random_state=5
)

classifier = make_pipeline(
    StandardScaler(),
    SVC(kernel="poly", degree=3, C=1.0, gamma="scale", max_iter=50000)
)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Polynomial SVM")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Precision:", round(precision_score(y_test, y_pred, average="weighted") * 100, 2), "%")
print("Recall:", round(recall_score(y_test, y_pred, average="weighted") * 100, 2), "%")
print("F1:", round(f1_score(y_test, y_pred, average="weighted") * 100, 2), "%")