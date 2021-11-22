import numpy as np
import cv2 as cv
import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
# from PIL import Image

# initial process - retrieving data
dataset = []
labels = []
classes = 43

for i in range(classes):
    path = os.path.join(os.getcwd(), 'GTSRB\\Train', str(i))
    images = os.listdir(path)
    
    for j in images:
        try:
            # image = Image.open(path + '\\' + j)
            # image = image.resize((30, 30))
            # image_array = np.array(image)
            # dataset.append(image_array)
            # labels.append(i)
            image = cv.imread(path + '\\' + j)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = cv.resize(image, dsize = (30, 30), interpolation = cv.INTER_CUBIC)
            dataset.append(image)
            labels.append(i)
        except:
            print("Error loading image")

dataset_array = np.array(dataset)
labels_array = np.array(labels)

print(dataset_array.shape, labels_array.shape)

# splitting dataset into training model and test model
X_train, X_test, y_train, y_test = train_test_split(dataset_array, labels_array, test_size = 0.2, random_state = 68)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# building the model
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5, 5), activation = 'relu', input_shape = X_train.shape[1:]))
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
model.add(Dense(43, activation = 'softmax'))

# model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# y_train = to_categorical(y_train, 43)
# y_test = to_categorical(y_test, 43)
history = model.fit(X_train, y_train, batch_size = 32, epochs = 2, validation_data = (X_test, y_test))
model.save("TSR_model.h5")

# plotting
plt.figure(0)
plt.plot(history.history['accuracy'], label = 'training accuracy')
plt.plot(history.history['val_accuracy'], label = 'val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()