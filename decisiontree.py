
#this code is for decision tree

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   
import pandas as pd
from sklearn import metrics 
import sklearn.metrics  
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split    
from sklearn.metrics import accuracy_score  
from sklearn.metrics import classification_report   
from sklearn import tree    
from sklearn.tree import export_graphviz    
  
from sklearn import preprocessing   
from sklearn.preprocessing import LabelEncoder  
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import StandardScaler    
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer



#load data

file= pd.read_csv('C:/Users/Usuario/Documents/GitHub/HENRY-PROYECTO-FINAL/doc/iris.csv ')
print(file.head(3))
print(file.shape)

#split data

X = file.values[:, 0:4]
Y = file.values[:,4]
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

#train model

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=5)

clf_gini.fit(X_train, y_train)

#predict

y_pred = clf_gini.predict(X_test)



#evaluate
# Accuracy
#   Accuracy is the most intuitive performance measure and it is simply a ratio of correctly predicted observation to the total observations.


print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)

# Precision
#   Precision is the ratio of correctly predicted positive observations to the total predicted positive observations.

print ("Precision is ", metrics.precision_score(y_test, y_pred, average='weighted')*100)

# Recall
#   Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes.

print ("Recall is ", metrics.recall_score(y_test, y_pred, average='weighted')*100)

# F1 score
#   F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account.

print ("F1 score is ", metrics.f1_score(y_test, y_pred, average='weighted')*100)

# Confusion Matrix
#   Confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known.

print ("Confusion Matrix is ", metrics.confusion_matrix(y_test, y_pred))

# Classification Report

print ("Classification Report is ", metrics.classification_report(y_test, y_pred))

#visualize

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf_gini,
                     feature_names=file.columns[0:4],
                        class_names=file.columns[4],
                        filled=True)

plt.show()
#save model

tree.export_graphviz(clf_gini, out_file='tree.dot', feature_names=file.columns[0:4], class_names=file.columns[4], filled=True)

