from sklearn.datasets import load_wine
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

"""
Simple Decision Tree Classifier that identifies types of Wine.
@Author Afaq Anwar
@Version 02/25/2019
"""

# Sets up the data as a DataFrame in order to easier work with data.
df = pd.DataFrame(load_wine().data)
df.columns = load_wine().feature_names
df['type'] = load_wine().target

# X = Features, y = labels
X = df.drop('type', axis=1)
y = df['type']

# Splits the data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

print(accuracy_score(y_test, predictions))

# Visualization code.
import graphviz
dot_data = tree.export_graphviz(classifier, out_file=None,
                                feature_names=load_wine().feature_names,
                                class_names=load_wine().target_names,
                                filled=True, rounded= True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("wine")
