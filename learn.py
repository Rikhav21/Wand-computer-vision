import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
def load_data(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            image = cv2.imread(os.path.join(folder, filename))
            image = cv2.resize(image, (100, 100))
            images.append(image)
            labels.append(folder)
    return images, labels
lumoses, lumoslables = load_data("lumos")
noxesimg, noxes = load_data("nox")
noimg, nolables = load_data("none")
images = lumoses + noxesimg + noimg
labels = lumoslables + noxes + nolables
images = np.array(images)
labels = np.array(labels)
endcoder = LabelEncoder()
encodes = endcoder].fit_transform(labels)
Xtrain = Xtrain / 255.0
Xtest = Xtest / 255.0
ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(Xtest, ytest))
loss, accuracy = model.evaluate(Xtest, ytest)
print("Test Accuracy:", accuracy)
model.save("image_classifier_model.keras")
