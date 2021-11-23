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

    # error = []
    # score = {}
    # score_list = []
    # ##let's first calculate the mean of error for all the predicted values where K ranges from 1 and 40.
    # # Calculating error for K values between 1 and 40
    # for i in range(10, 50):
    #     knn = KNeighborsClassifier(n_neighbors=i)
    #     knn.fit(X_train, y_train)
    #     pred_i = knn.predict(X_test)
    #     error.append(np.mean(pred_i != y_test))
    #     score[i] = metrics.accuracy_score(y_test, pred_i)
    #     score_list.append(metrics.accuracy_score(y_test, pred_i))

    # plt.figure(figsize=(12, 6))
    # plt.plot(range(10, 50), error, color='blue', linestyle='dashed', marker='o',
    #          markerfacecolor='red', markersize=10)
    # plt.title('Error Rate K Value')
    # plt.xlabel('K Value')
    # plt.ylabel('Mean Error')
    # plt.show()

    # plt.plot(range(10,50), score_list)
    # plt.xlabel("value of k")
    # plt.ylabel("Testing Accuracy")
    # plt.show()