import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

image_dir = "dataset/"

no_tumor = os.listdir(image_dir + "no/")
yes_tumor = os.listdir(image_dir + "yes/")

dataset = []
label = []

INPUT_SIZE = 64

#
for i, image_name in enumerate(no_tumor):
    if image_name.split(".")[1] == "jpg":
        image = cv2.imread(image_dir + "no/" + image_name)
        image = Image.fromarray(image, "RGB")
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

#
for i, image_name in enumerate(yes_tumor):
    if image_name.split(".")[1] == "jpg":
        image = cv2.imread(image_dir + "yes/" + image_name)
        image = Image.fromarray(image, "RGB")
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(
    dataset, label, test_size=0.2, random_state=0
)

x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

y_train = keras.utils.to_categorical(y_train, num_classes=2)
y_test = keras.utils.to_categorical(y_test, num_classes=2)

# Model Building

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))


model.add(keras.layers.Conv2D(32, (3, 3), kernel_initializer="he_uniform"))
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))


model.add(keras.layers.Conv2D(64, (3, 3), kernel_initializer="he_uniform"))
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))


model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(2))
model.add(keras.layers.Activation("softmax"))


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(
    x_train,
    y_train,
    batch_size=16,
    verbose=True,
    epochs=30,
    validation_data=(x_test, y_test),
    shuffle=False,
)

model.save("BrainTumor10EpochsCaterogical.h5")
