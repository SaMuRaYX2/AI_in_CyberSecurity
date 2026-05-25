import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler

companies = np.array([
    "Apple", "Microsoft", "Google", "Amazon", "Meta",
    "Tesla", "Ford", "General Motors",
    "JPMorgan", "Bank of America", "Citigroup",
    "ExxonMobil", "Chevron"
])

np.random.seed(42)

tech = np.random.normal(loc=0.8, scale=0.2, size=(5, 10))
auto = np.random.normal(loc=-0.2, scale=0.2, size=(3, 10))
banks = np.random.normal(loc=0.2, scale=0.2, size=(3, 10))
energy = np.random.normal(loc=0.5, scale=0.2, size=(2, 10))

X = np.vstack([tech, auto, banks, energy])

X = StandardScaler().fit_transform(X)

model = AffinityPropagation(random_state=0)
model.fit(X)

labels = model.labels_

num_clusters = len(np.unique(labels))

print("Number of clusters:", num_clusters)

for i in range(num_clusters):
    print("Cluster", i + 1, "=>", ", ".join(companies[labels == i]))