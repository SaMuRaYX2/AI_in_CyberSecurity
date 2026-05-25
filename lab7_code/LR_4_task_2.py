import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

iris = load_iris()

X = iris.data
y = iris.target

kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=0)

kmeans.fit(X)

labels = kmeans.predict(X)

centers = kmeans.cluster_centers_

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap="viridis")
plt.scatter(centers[:, 0], centers[:, 1], c="black", s=200, alpha=0.7, marker="X")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("K-Means clustering for Iris dataset")
plt.grid(True)
plt.show()

print("Центри кластерів:")
print(centers)

print("Adjusted Rand Index:")
print(round(adjusted_rand_score(y, labels), 4))