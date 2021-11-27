'''
Sequential.py - A sequential neural network using the Keras package. 

https://www.bmc.com/blogs/keras-neural-network-classification/
'''

from keras.models import Sequential
from keras.layers import Dense

def processSequential(X_train, X_test, y_train, y_test):
    model = Sequential()

    #Change the layers and hidden units (first argument) during testing
    #https://keras.io/guides/sequential_model/
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #Experiement with different arguments during testing
    model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
    
    
    model.fit(X_train, y_train, epochs=150, batch_size=1, verbose=1)

    predictions = model.predict_classes(X_train)

    for i in (len(X_train)):
        print('%s => %d (expected %d)' % (X_train[i].tolist(), predictions[i], y_train[i]))