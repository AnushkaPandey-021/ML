import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix

# Load the Iris dataset
iris = load_iris()
X, y = iris.data[:, :2], iris.target

# Initialize and fit K-Means clustering
classifier = KMeans(n_clusters=len(np.unique(y)), random_state=42)
classifier.fit(X)

# Predict cluster labels and map them to class labels
cluster_labels = classifier.predict(X)
cluster_class_labels = np.array([np.bincount(y[cluster_labels == i]).argmax() for i in range(len(np.unique(y)))])
y_pred = cluster_class_labels[cluster_labels]

# Evaluate the classifier's performance
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred))

# Visualize the dataset and cluster centers
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label='Data')
plt.scatter(classifier.cluster_centers_[:, 0], classifier.cluster_centers_[:, 1], c='red', marker='o', s=100, label='Cluster Centers')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('K-Means Clustering on Iris Dataset')
plt.legend()
plt.show()
