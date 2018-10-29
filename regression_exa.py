# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 21:10:20 2018

@author: RandySteven
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
import seaborn as sns; sns.set()
# special matplotlib argument for improved plots
import matplotlib.pyplot as plt
import statsmodels.api as sm

#pd.set_option('display.expand_frame_repr', False)

sns.set_style("whitegrid")

boston = load_boston()

print(boston.data.shape)

print(boston.DESCR)

bos = pd.DataFrame(boston.data)
bos.columns = boston.feature_names

bos['PRICE'] = boston.target
print(bos.head())
print(bos.describe())

X = bos.drop('PRICE', axis = 1)
Y = bos['PRICE']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 8)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

lm = LinearRegression(fit_intercept=True)
lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)

plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.plot([0, 50], [0, 50], '--k')

mse = mean_squared_error(Y_test, Y_pred)

print("Model slope:    ", lm.coef_[0])
print("Model intercept:", lm.intercept_)
print(lm.score(X,Y))
print(mse)
print("RMS: %r " % np.sqrt(np.mean((Y_pred - Y_test) ** 2)))

X = bos["RM"]
y = bos['PRICE']

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
print(model.summary())

X = bos["RM"]
y = bos['PRICE']
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
print(model.summary())

X = bos
y = bos['PRICE']
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
print(model.summary())