'''
kNN.py - Model application and evaluation for K-Nearest Neighbors
'''
import metrics as metrics

def processKNN(X_train, X_test, y_train, y_test):


    #Apply kNN
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=15)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    metrics.printMetrics("K-NEAREST NEIGHBORS", y_test, y_pred)
    