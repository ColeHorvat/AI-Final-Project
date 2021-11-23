import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy
import os
import sys
import pickle
import librosa
import librosa.display
from IPython.display import Audio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

import models.kNN as kNN
import models.DecisionTree as DecisionTree
import models.GaussianBayes as GaussianBayes
import models.RandomForests as RandomForests

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


kNN.processKNN(X_train, X_test, y_train, y_test)
DecisionTree.processDecisionTree(X_train, X_test, y_train, y_test)
GaussianBayes.processGaussianBayes(X_train, X_test, y_train, y_test)
RandomForests.processRandomForest(X_train, X_test, y_train, y_test)