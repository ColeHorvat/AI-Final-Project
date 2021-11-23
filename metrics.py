'''
metrics.py - This is where all the code for getting model metrics will be written
'''

#import matplotlib.pyplot as plt <-- MAY NEED LATER
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

def printMetrics(modelTitle, y_test, y_pred):
    #Confusion Matrix and Classification Report
    print(modelTitle + "\n")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))