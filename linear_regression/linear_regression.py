import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import psutil

def print_memory_usage(message):
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"{message} - RSS: {memory_info.rss / (1024 ** 2):.2f} MB")

print_memory_usage("Before loading dataset")
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Divide the dataset into training (90%) and validation (10%)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=42)

# Train a LASSO model with a penalty of 0.3
print_memory_usage("Before training the model")
lasso = Lasso(alpha=0.3)
lasso.fit(X_train, y_train)
print_memory_usage("After training the model")

# Print the intercept and coefficients of the model
#print(f"Intercept: {lasso.intercept_}")
#print(f"Coefficients: {lasso.coef_}")

# Calculate R² on the validation set
print_memory_usage("Before prediction")
y_pred = lasso.predict(X_valid)
print_memory_usage("After prediction")
r2 = r2_score(y_valid, y_pred)
print(f"R²: {r2}")

# Calculate the mean squared error (MSE) on the validation set
mse = mean_squared_error(y_valid, y_pred)
print(f"Mean Squared Error: {mse}")
