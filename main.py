'''
main.py - This is the main script where all the models will be called
'''

#Package Imports
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#Model Imports
import models.kNN as kNN
import models.DecisionTree as DecisionTree
import models.GaussianBayes as GaussianBayes
import models.RandomForests as RandomForests
import models.Sequential as Sequential

TEST_SIZE = 0.33

#Read features_30_sec.csv
csv = pd.read_csv("./Data/features_30_sec.csv")

#Preprocess genre label data
labelList = csv.iloc[:,-1]
encoder = LabelEncoder()

#y is the encoded list of genre labels
y = encoder.fit_transform(labelList)

#X is the rest of the data other than genre, standardized to avoid
#non-proportional variance
scalar = StandardScaler()
X = scalar.fit_transform(np.array(csv.iloc[:,1:-1], dtype=float))


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=TEST_SIZE)


# kNN.processKNN(X_train, X_test, y_train, y_test)
# DecisionTree.processDecisionTree(X_train, X_test, y_train, y_test)
# GaussianBayes.processGaussianBayes(X_train, X_test, y_train, y_test)
# RandomForests.processRandomForest(X_train, X_test, y_train, y_test)
Sequential.processSequential(X_train, X_test, y_train, y_test)