'''
DecisionTree.py - Model application and evaluation for Decision Trees
'''

import metrics as metrics
def processDecisionTree(X_train, X_test, y_train, y_test):
    
    #Apply Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    #Evaluate Decision Tree
    metrics.printMetrics("DECISION TREE", y_test, y_pred)