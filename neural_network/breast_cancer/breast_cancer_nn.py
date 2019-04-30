from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

"""
Feed Forward Neural Network that classifies breast cancer as malignant or benign.
@Author Afaq Anwar
@Version 02/26/2019
"""

# Sets up Data in order to be easy to use.
df = pd.DataFrame(load_breast_cancer().data)
df.columns = load_breast_cancer().feature_names
df['type'] = load_breast_cancer().target

# X = Features, y = Labels
X = df.drop('type', axis=1)
y = df['type']

# Splits the data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Since we are working with a feed forward Neural Network scaled data is required.
scaler = StandardScaler()
scaler.fit(X_train)

"""
The hidden layers were chosen arbitrarily there is really no thought process behind this besides
keeping it between the number of inputs and outputs. 
"""
mlp = MLPClassifier(hidden_layer_sizes=(18, 12), max_iter=1000)
mlp.fit(X_train, y_train)

# Predicts the labels of the test data.
predictions = mlp.predict(X_test)

# Prints the accuracy.
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
