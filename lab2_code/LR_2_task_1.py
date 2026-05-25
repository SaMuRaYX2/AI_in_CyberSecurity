import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

input_file = "income_data.txt"

X = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, "r") as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break

        if "?" in line:
            continue

        data = line.strip().split(", ")

        if data[-1] == "<=50K" and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1

        elif data[-1] == ">50K" and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)

label_encoder = []
X_encoded = np.empty(X.shape)

for i, item in enumerate(X[0]):
    if item.replace("-", "").isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoder.append(le)

X_data = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y, test_size=0.2, random_state=5
)

classifier = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=10000))
classifier.fit(X_train, y_train)

y_test_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average="weighted")
recall = recall_score(y_test, y_test_pred, average="weighted")
f1 = f1_score(y_test, y_test_pred, average="weighted")

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1 score:", round(f1 * 100, 2), "%")

print("\nClassification report:")
print(classification_report(y_test, y_test_pred))

cv_f1 = cross_val_score(classifier, X_data, y, scoring="f1_weighted", cv=3)
print("Cross-validation F1:", round(100 * cv_f1.mean(), 2), "%")

input_data = [
    "37", "Private", "215646", "HS-grad", "9", "Never-married",
    "Handlers-cleaners", "Not-in-family", "White", "Male",
    "0", "0", "40", "United-States"
]

input_data_encoded = []
encoder_index = 0

for item in input_data:
    if item.replace("-", "").isdigit():
        input_data_encoded.append(int(item))
    else:
        encoded_value = label_encoder[encoder_index].transform([item])[0]
        input_data_encoded.append(encoded_value)
        encoder_index += 1

input_data_encoded = np.array(input_data_encoded).reshape(1, -1)

predicted_class = classifier.predict(input_data_encoded)
result = label_encoder[-1].inverse_transform(predicted_class)

print("\nTest point prediction:", result[0])