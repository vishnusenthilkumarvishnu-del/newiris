from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
print("Dataset loaded successfully.")
print(f"Number of samples: {len(X)}")
print(f"Feature names: {iris.feature_names}")
print(f"Target names: {iris.target_names}")

import pandas as pd
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset summary:")
print(df.describe())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print("Model trained successfully.")

y_pred = knn.predict(X_test)
print("Predictions made on the test set.")
print("First 5 predictions:", y_pred[:5])
print("First 5 actual labels:", y_test[:5])

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

import matplotlib.pyplot as plt
plt.figure()
for species in range(3):
    plt.scatter(
        df[df['species'] == species]['sepal length (cm)'],
        df[df['species'] == species]['sepal width (cm)'],
        label=iris.target_names[species]
    )

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Iris Dataset - Sepal Length vs Width")
plt.legend()
plt.show()