from sklearn import tree

"""
Simple Decision Tree Classifier.
@Author Afaq Anwar
@Version 02/23/2019
"""

# Also sometimes represented as X
# [Weight (Grams), Smooth or Bumpy (1 = smooth, 0 = bumpy)
features = [[130, 1], [130, 1], [150, 0], [170, 0]]

# Also sometimes represented as y
# 0 = Apple, 1 = Orange
labels = [0, 0, 1, 1]

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features, labels)

# Should most likely result as being labeled an "Orange".
test_data = classifier.predict([[145, 0]])

if test_data[0] == 1:
    print("Orange")
elif test_data[0] == 0:
    print("Apple")
