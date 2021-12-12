import cv2
from tensorflow import keras
from PIL import Image
import numpy as np


model = keras.models.load_model("BrainTumor10EpochsCaterogical.h5")

image = cv2.imread(
    "C:\\Users\\ivand\\Documents\\BrainTumorDetector\\dataset\\pred\\pred2.jpg"
)

img = Image.fromarray(image)

img = img.resize((64, 64))

img = np.array(img)

input_img = np.expand_dims(img, axis=0)
predict = model.predict(input_img)
result = np.argmax(predict, axis=1)
print(result)
