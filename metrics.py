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

    #MATPLOTLIB.PYPLOT GRAPH EXAMPLE

    # import matplotlib.pyplot as plt
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