import numpy as np
import cv2 as cv
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense

def main():
    # Ask for which training set
    trainingset = input('What training set do you want to use? (GTSRB/BTSC): ')

    # Initial process - retrieving data
    dataset = []
    labels = []

    # Choose which dataset is used
    if trainingset.upper() == 'GTSRB':
        dataset_path = os.path.join(os.getcwd(), 'GTSRB\\Train')
    elif trainingset.upper() == 'BTSC':
        dataset_path = os.path.join(os.getcwd(), 'BTSC\\BelgiumTSC_Training\\Training')
    else:
        exit('No such training option!')

    # Count the classes of the input dataset
    classes = len([name for name in os.listdir(dataset_path)])

    # Load images
    for i in range(classes):
        path = os.path.join(dataset_path, str(i))
        images = os.listdir(path)
        
        for j in images:
            try:
                image = cv.imread(path + '\\' + j)
                # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                image = cv.resize(image, dsize = (30, 30), interpolation = cv.INTER_CUBIC)
                dataset.append(image)
                labels.append(i)
            except:
                print("Error loading image!")

    dataset_array = np.array(dataset)
    labels_array = np.array(labels)

    print(dataset_array.shape, labels_array.shape)

    # Splitting dataset into training model and test model
    X_train, X_test, y_train, y_test = train_test_split(dataset_array, labels_array, test_size = 0.2, random_state = 68)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    y_train = to_categorical(y_train, classes)
    y_test = to_categorical(y_test, classes)

    # Building the model
    model = Sequential()
    # model.add(Conv2D(filters = 32, kernel_size = (5, 5), activation = 'relu', input_shape = X_train.shape[1:])) # For RGB/BGR input
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), activation = 'relu', input_shape = (30, 30, 1))) # For grayscale input
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), activation = 'relu',))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(rate = 0.25))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(rate = 0.25))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(classes, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    # model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    model.fit(X_train, y_train, batch_size = 32, epochs = 3, validation_data = (X_test, y_test))
    model.save("models/TSR_model.h5")

if __name__ == "__main__":
    main()