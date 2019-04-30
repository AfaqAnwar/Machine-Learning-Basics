from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

"""
Simple classifier that determines the price of a house within the Boston market.
@Author Afaq Anwar
@Verison 02/24/2019
"""

# Data setup.
df = pd.DataFrame(load_boston().data)
df.columns = load_boston().feature_names
df['PRICE'] = load_boston().target

X = df.drop('PRICE', axis=1)
y = df['PRICE']

# Data split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

classifier = LinearRegression()
classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)

print(accuracy)
