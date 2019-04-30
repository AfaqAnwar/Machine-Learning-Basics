from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

"""
Simple classification of Iris flowers using the KNearestNeighbor Classifier
@Author Afaq Anwar
@Verison 02/25/2019
"""

iris_data = load_iris()

# Indexes of Iris flowers that will be used for testing the Classifier.
test_indexes = [0, 50, 100]

# Features
X = iris_data.data
# Labels
y = iris_data.target

# Splits the data in half for training and testing purposes.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

# All features need to be scaled when using a Neural Network or algorithms that measure distance.
scaler = StandardScaler()
scaler.fit(X_train)

classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

print(accuracy_score(y_test, predictions))