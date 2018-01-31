import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

df = pd.read_csv("D:/Work/datascience-ws/dataset/kaggle/titanic/train.csv")
#df = pd.read_csv("dataset/train.csv")
#print(df.head())

features = np.array(['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])
#clf = RandomForestClassifier()
clf = RandomForestClassifier(max_depth=2, random_state=0)
#could not convert string to float: 'Q'  <-- https://stackoverflow.com/questions/30384995/randomforestclassfier-fit-valueerror-could-not-convert-string-to-float
# need to do labelencoder or oneHotEncodershow()
clf.fit(df[features],df['Survived'])

importances = clf.feature_importances_
sorted_idx = np.argsort(importances)

padding = np.arange(len(features)) + 0.5
plt.barh(padding, importances[sorted_idx], align='center')
plt.yticks(padding, features[sorted_idx])
plt.xlabel("Relative Importance")
plt.title("Variable Importance")
