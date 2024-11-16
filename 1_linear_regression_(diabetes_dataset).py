import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Create and train the linear regression model
classifier = linear_model.LinearRegression()
classifier.fit(X_train, y_train)

# Predict values for X_test data
y_pred = classifier.predict(X_test)

# Print regression coefficients, intercept, and variance score
print(f'Coefficients: {classifier.coef_}')
print(f'Intercept: {classifier.intercept_}')
print(f'Variance score: {classifier.score(X_test, y_test)}')

# Print mean squared error
print(f'Mean squared error: {mean_squared_error(y_test, y_pred):.2f}')

# Plot expected vs predicted values
plt.title('Linear Regression (Diabetes Dataset)')
plt.scatter(y_test, y_pred, c='b', marker='.', s=36)
plt.plot([0, 330], [0, 330], '--r', linewidth=2)
plt.xlabel('Expected')
plt.ylabel('Predicted')
plt.show()
