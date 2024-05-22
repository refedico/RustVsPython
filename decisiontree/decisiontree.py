import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import psutil

def print_memory_usage(message):
    process = psutil.Process()
    rss = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
    print(f"{message} - RSS: {rss:.2f} MB")

# Monitoraggio memoria prima del caricamento del dataset
print_memory_usage("Before loading dataset")

# Carica il dataset Iris
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Dividi il dataset in train e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Monitoraggio memoria dopo il caricamento del dataset
print_memory_usage("After loading dataset")

print("Training model with Gini criterion ...")
print_memory_usage("Before training Gini model")

# Addestra un modello Decision Tree con criterio di Gini
gini_model = DecisionTreeClassifier(criterion='gini', max_depth=100, min_samples_split=2, min_samples_leaf=1)
gini_model.fit(X_train, y_train)

# Monitoraggio memoria dopo l'addestramento del modello Gini
print_memory_usage("After training Gini model")

# Predici i risultati sul set di test
gini_pred_y = gini_model.predict(X_test)

# Calcola la matrice di confusione e l'accuratezza
cm = confusion_matrix(y_test, gini_pred_y)
accuracy = accuracy_score(y_test, gini_pred_y)

print(cm)
print(f"Test accuracy with Gini criterion: {accuracy * 100:.2f}%")

# Stampa le feature importances
feats = gini_model.feature_importances_
print(f"Features importances: {feats}")

