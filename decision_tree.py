# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 10:41:20 2018
@author: FUN13
"""
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.model_selection import cross_val_score,KFold,train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

plt.close("all")
sns.set()

# load dataset from sklearn
data = load_iris()

## Convert loaded dataset into dataframe
#df = pd.DataFrame(data= np.c_[data['data'], data['target']],
#                     columns= data['feature_names'] + ['target'])

## Get dataset from url
#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

## load dataset from .csv 
df = pd.read_csv('Iris.csv')

## Get data
# feature matrix
X = data.data

# target vector


# class labels
labels = data.feature_names

#Species
#Species = data.target_names
Species=df.Species.unique().tolist()

# print dataset description
print(data.DESCR)

#Data exploration
print(df.isnull().any())

print(df.dtypes)

print(df.describe())

df['PetalWidthCm'].plot.hist()
plt.show()

sns.pairplot(df,hue='Species')

measures=list(df)[1:]
piris = pd.melt(df[measures], "Species", var_name="measurement") 
sns.factorplot(x="measurement", y="value", hue="Species", data=piris, size=7, kind="bar",palette="bright") 
plt.show() 
print(piris.head())

## Classification

all_inputs = df[measures[:-1]].values
all_classes = df['Species'].values

(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=1)

print('There are {} samples in the training set and {} samples in the test set'.format(train_inputs.shape[0], test_inputs.shape[0]))

dtc = DecisionTreeClassifier()
dtc.fit(train_inputs, train_classes)

graph = Source(tree.export_graphviz(dtc, out_file=None
   , feature_names=labels, class_names=df['Species'].unique().tolist()
   , filled = True))
display(SVG(graph.pipe(format='svg')))

print(dtc)


print('The accuracy of the Decision Tree classifier on training data is {:.2f}'.format(dtc.score(train_inputs, train_classes)))
print('The accuracy of the Decision Tree classifier on test data is {:.2f}'.format(dtc.score(test_inputs, test_classes)))


## Regression
enc = OneHotEncoder(handle_unknown='ignore')

regression_tree2 = DecisionTreeRegressor(max_depth=3,random_state=0)
regression_tree4 = DecisionTreeRegressor(max_depth=4,random_state=0)

enc.fit(train_classes.reshape(-1,1))

transformed_train=0+enc.transform(train_classes.reshape(-1,1)).toarray()

regression_tree2.fit(train_inputs,transformed_train)
regression_tree4.fit(train_inputs,transformed_train)

# Predict
y_1 = regression_tree2.predict(test_inputs)
y_2 = regression_tree4.predict(test_inputs)

# Plot the results

colorplot=dict(zip(list(enc.categories_)[0].tolist(),['r','b','g']))
#colorplot=dict(zip([0,1,2],['r','b','g']))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig)

plt.title("Decision Tree Regression MD 3")
Species_r=[x[0] for x in enc.inverse_transform(np.rint(y_1)).tolist()]
for a,color in zip (test_inputs,Species_r):
    ax.scatter(a[0],a[1],a[2],color=colorplot[color])
    
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig)

plt.title("Decision Tree Regression MD 3")
Species_r=[x[0] for x in enc.inverse_transform(np.rint(y_2)).tolist()]
for a,color in zip (test_inputs,Species_r):
    ax.scatter(a[0],a[1],a[2],color=colorplot[color])
    
print("The norm of the errors is: "+str(np.linalg.norm(np.rint(y_1)-y_2)))

crossvalidation = KFold(5,shuffle=True, random_state=1)

y=all_classes
enc.fit(y.reshape(-1,1))
y=enc.transform(y.reshape(-1,1)).toarray()

score = np.mean(cross_val_score(regression_tree2,X, y,scoring='r2', cv=crossvalidation
                                ,n_jobs=1))
print ('Mean squared error: %.3f' % abs(score))
score = np.mean(cross_val_score(regression_tree4,X, y,scoring='r2', cv=crossvalidation
                                ,n_jobs=1))
print ('Mean squared error: %.3f' % abs(score))

score = np.mean(cross_val_score(regression_tree2,df[measures[:-1]].values, y,scoring='r2', cv=crossvalidation
                                ,n_jobs=1))
print ('Mean squared error: %.3f' % abs(score))
score = np.mean(cross_val_score(regression_tree4,df[measures[:-1]].values, y,scoring='r2', cv=crossvalidation
                                ,n_jobs=1))
print ('Mean squared error: %.3f' % abs(score))