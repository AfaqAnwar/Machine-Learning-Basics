from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

"""
MY FIRST NEURAL NETWORK!
This classifies hand written digits (8x8 images).
@Author Afaq Anwar
@Version 02/25/2019
"""

# Sets up Data in order to be easy to use.
df = pd.DataFrame(load_digits().data)
df['digit'] = load_digits().target

# X = Features, y = Labels
X = df.drop('digit', axis=1)
y = df['digit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Scales each feature to a scale of 0-1.
# Since we are working with a feed forward Neural Network scaled data is required.
scaler = StandardScaler()
scaler.fit(X_train)

# Creates the Neural Network with 2 hidden layers of 10 neurons.
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)

# Prints the accuracy.
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))