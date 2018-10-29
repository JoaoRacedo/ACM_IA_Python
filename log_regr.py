# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 23:29:17 2018

@author: RandySteven
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


diabetes = pd.read_csv('diabetes.csv')
print(diabetes.columns)

print(diabetes.groupby('Outcome').size())

columns=list(diabetes)

sns.countplot(diabetes['Outcome'],label="Count")

diabetes.info()

corr = diabetes.corr()
fig=plt.figure()
print(corr)
sns.heatmap(corr,xticklabels=corr.columns,
        yticklabels=corr.columns)

table1=np.mean(diabetes,axis=0)
table2=np.std(diabetes,axis=0)
print(table1)
print(table2)

a=columns[0:3]+(columns[4:])

X_train,X_test,y_train,y_test=train_test_split(diabetes[a[:-1]],diabetes['Outcome'],test_size=0.25,random_state=0)

inputData=X_train
outputData=y_train


logit1=LogisticRegression()
logit1.fit(inputData,outputData)
logit1.score(inputData,outputData)

##True positive
trueInput=diabetes.ix[diabetes['Outcome']==1].iloc[:,:8]
trueOutput=diabetes.ix[diabetes['Outcome']==1].iloc[:,8]
##True positive rate
np.mean(logit1.predict(trueInput[a[:-1]])==trueOutput)
##Return around 55%

##True negative
falseInput=diabetes.ix[diabetes['Outcome']==0].iloc[:,:8]
falseOutput=diabetes.ix[diabetes['Outcome']==0].iloc[:,8]
##True negative rate
np.mean(logit1.predict(falseInput[a[:-1]])==falseOutput)
##Return around 90%

###Confusion matrix with sklearn

print(confusion_matrix(logit1.predict(inputData),outputData))


y_pred=logit1.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
##Computing false and true positive rates
fpr, tpr,_=roc_curve(logit1.predict(inputData),outputData,drop_intermediate=False)


plt.figure()
##Adding the ROC
plt.plot(fpr, tpr, color='red',
 lw=2, label='ROC curve')
##Random FPR and TPR
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
##Title and label
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print(roc_auc_score(logit1.predict(inputData),outputData))

plt.figure()
plt.scatter(inputData.iloc[:,1],inputData.iloc[:,5],c=logit1.predict_proba(inputData)[:,1],alpha=0.4)
plt.xlabel('Glucose level ')
plt.ylabel('BMI ')
plt.title("Predicted6")
plt.show()

plt.figure()
plt.scatter(inputData.iloc[:,1],inputData.iloc[:,5],c=outputData,alpha=0.4)
plt.xlabel('Glucose level ')
plt.ylabel('BMI ')
plt.show()

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

plt.figure()
y_pred_proba = logit1.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()