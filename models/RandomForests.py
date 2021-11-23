import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy
import os
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

import metrics as metrics
def processRandomForest(X_train, X_test, y_train, y_test):
    
    #Apply Random Forest
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(max_depth=5)#EXPERIMENT WITH CHANGING max_depth
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    #Evaluate Random Forest
    metrics.printMetrics("RANDOM FOREST", y_test, y_pred)