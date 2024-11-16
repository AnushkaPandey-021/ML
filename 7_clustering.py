import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Hierarchical Clustering and Dendrograms
linkage_methods = ['ward', 'single', 'complete']
plt.figure(figsize=(15, 5))

for i, method in enumerate(linkage_methods):
    labels = AgglomerativeClustering(n_clusters=3, linkage=method).fit_predict(X)
    plt.subplot(1, 3, i + 1)
    dendrogram(linkage(X, method=method), labels=labels)
    plt.title(f"{method.capitalize()} Linkage")

# Clustering Results
plt.figure(figsize=(15, 5))
for i, method in enumerate(linkage_methods):
    labels = AgglomerativeClustering(n_clusters=3, linkage=method).fit_predict(X)
    plt.subplot(1, 3, i + 1)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title(f"{method.capitalize()} Linkage")

# Feature Engineering and Classification
X_with_cluster = np.column_stack((X, AgglomerativeClustering(n_clusters=3, linkage='complete').fit_predict(X)))
X_train, X_test, y_train, y_test = train_test_split(X_with_cluster, y, test_size=0.2, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

plt.show()
