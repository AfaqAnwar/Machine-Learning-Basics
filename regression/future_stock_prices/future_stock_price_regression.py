import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use("ggplot")

"""
Linear Regression Introduction with Python, scikit-learn, Pandas, Numpy, and Matplotlib.
Predicts Google's Stock ~30 out with ~98% accuracy.
This is not actual model that can be used because it's essentially a "simulator" since all data is just shifted up,
resulting in null data which then is predicted.
@Author Afaq Anwar
@Version 02/16/2019
"""

# Authenticate API KEY with Quandl
quandl.ApiConfig.api_key = "Z-DU4oT-cFzqdsq5E2oM"

# Obtains the EOD Stock Prices for Microsoft.
df = quandl.get("EOD/MSFT")

# Obtains the specific columns from the data set.
df = df[['Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Adj_Volume']]

# Creates a new column High Low Percent Change that shows stock volatility.
df['HL_PCT'] = (df['Adj_High'] - df['Adj_Low']) / df['Adj_Low'] * 100

# Creates a new column Percent Change that shows the change of the Stock within the current trading day.
df['PCT_Change'] = (df['Adj_Close'] - df['Adj_Open']) / df['Adj_Open'] * 100

# Obtains all features needed for future use.
# All features are attributes of a potential cause in future price.
df = df[['Adj_Close', 'HL_PCT', 'PCT_Change', 'Adj_Volume']]

future_cast_col = "Adj_Close"

# ML cannot operate with no data, thus setting all null data to an outlier will help.
# This allows us to use the column's existing data.
df.fillna(-99999, inplace=True)

# Predicts out 0.35% of the data.
# math.ceil is used to round up to the nearest whole day.
forecast_out = int(math.ceil(0.0035*len(df)))

"""
Creates our label, shifts the data in the rows above,
thus allowing the data in the future_cast_col to represent data within the future.

Shifting positively ships the row values downwards, shifting negatively shifts the row values upwards.
"""
df['Label'] = df[future_cast_col].shift(-forecast_out)

# X represents Features, Y represents Labels.
# Features ; Are everything besides the Label column.
# In this case the Label is the future Adjusted Close price.
X = np.array(df.drop(['Label'], 1))

# Scales all the data down for easier use. In practical usage, all data should be scaled not just some of it.
X = preprocessing.scale(X)

# Future Features to be predicted.
X_future = X[-forecast_out:]

# Current Features to train with.
X = X[:-forecast_out]

# Removes all null data. Since we are predicting for all the data that doesn't have a Y value.
df.dropna(inplace=True)

y = np.array(df['Label'])

# Sets training & testing data. 20% of the data will be testing data. Data is shuffled.
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Creates a new Linear Regression Classifier.
# This can easily be changed with the Support Vector Regression Classifier.
# n_jobs defines the threads the Linear Regression Algorithm is using, it can use multiple unlike a SVR.
# classifier = LinearRegression(n_jobs=-1)

# Fits the Classifier with features and labels.
# classifier.fit(X_train, y_train)

# Saves the trained model.
# with open('linearregression.pickle', 'wb') as f:
#    pickle.dump(classifier, f)

# Loads the trained model.
pickle_input = open('linearregression.pickle', 'rb')
classifier = pickle.load(pickle_input)

# Scores the Classifier on testing data, previously unseen data.
accuracy = classifier.score(X_test, y_test)

forecast_set = classifier.predict(X_future)
print(forecast_set, accuracy, forecast_out)

"""
Matploit Lib Boiler Plate Code to generate a graph.
"""
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    # Finds the index based on dates. All data besides the Forecast is null, since those values are not obtainable.
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df["Adj_Close"].plot()
df["Forecast"].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("")
plt.show()
