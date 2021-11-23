'''
RandomForests.py - Model application and evaluation for Random Forests
'''

import metrics as metrics
def processRandomForest(X_train, X_test, y_train, y_test):
    
    #Apply Random Forest
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(max_depth=5)#EXPERIMENT WITH CHANGING max_depth
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    #Evaluate Random Forest
    metrics.printMetrics("RANDOM FOREST", y_test, y_pred)