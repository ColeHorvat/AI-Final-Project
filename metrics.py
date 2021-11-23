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

def printMetrics(modelTitle, y_test, y_pred):
    #Confusion Matrix and Classification Report
    print(modelTitle + "\n")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))