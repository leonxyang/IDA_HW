import pandas as pd

import numpy as np
import random

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt

df2 = pd.read_csv("Biomechanical_Data_2Classes.csv")
df2_features = df2.drop("class", axis = 1)
df2_class =df2["class"]

df2_xtrain, df2_xtest, df2_ytrain, df2_ytest = train_test_split(df2_features, df2_class, test_size = 80, random_state = 1)

clf_list = []
df2_ypred_list = []
accu_list = []
i = 0
for min_per_leaf in [5, 15, 25, 40, 50]:
    clf_list.append(tree.DecisionTreeClassifier(min_samples_leaf= min_per_leaf))
    clf_list[i].fit(df2_xtrain, df2_ytrain)
    df2_ypred_list.append(clf_list[i].predict(df2_xtest))
    accu_list.append(metrics.accuracy_score(df2_ytest, df2_ypred_list[i]))
    i += 1

#print(accu_list)

p_r_f_normal = []
for i in range(0, 5):
    p_r_f_normal.append(metrics.precision_recall_fscore_support(df2_ytest, df2_ypred_list[i], average=None, labels=["Normal"]))

p_r_f_abnormal = []
for i in range(0, 5):
    p_r_f_abnormal.append(metrics.precision_recall_fscore_support(df2_ytest, df2_ypred_list[i], average=None, labels=["Abnormal"]))

p_normal = []
for i in range(0, 5):
    p_normal.append(p_r_f_normal[i][0][0])

p_abnormal = []
for i in range(0, 5):
    p_abnormal.append(p_r_f_abnormal[i][0][0])

r_normal = []
for i in range(0, 5):
    r_normal.append(p_r_f_normal[i][1][0])

r_abnormal = []
for i in range(0, 5):
    r_abnormal.append(p_r_f_abnormal[i][1][0])



#print(p_r_f_normal[0][0][0])
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(df2_xtrain, df2_ytrain)
#df2_ypred = clf.predict(df2_xtest)
#accu = metrics.accuracy_score(df2_ytest, df2_ypred)


