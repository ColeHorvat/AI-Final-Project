'''
Gaussian Bayes.py - Model application and evaluation for Gaussian Bayes
'''

import metrics as metrics
def processGaussianBayes(X_train, X_test, y_train, y_test):
    
    #Apply Gaussian Bayes
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    #Evaluate Gaussian Bayes
    metrics.printMetrics("GAUSSIAN BAYES", y_test, y_pred)