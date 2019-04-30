from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
Simple classification of Iris flowers using a Decision Tree.
@Author Afaq Anwar
@Verison 02/24/2019
"""

iris_data = load_iris()

# Features
X = iris_data.data
# Labels
y = iris_data.target

# Splits the data in half for training and testing purposes.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

print(accuracy_score(y_test, predictions))

# Visualization Code
import graphviz
dot_data = tree.export_graphviz(classifier, out_file=None,
                     feature_names=iris_data.feature_names,
                     class_names=iris_data.target_names,
                     filled=True, rounded=True,
                     special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris")
