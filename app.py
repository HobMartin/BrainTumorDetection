import os
from flask import Flask, request, render_template
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = keras.models.load_model("BrainTumor10EpochsCaterogical.h5")
print("Model loaded")


def get_class_name(result_code):
    if result_code == 0:
        return "Немає пухлини"
    elif result_code == 1:
        return  "Є Пухлина"


def get_result(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, "RGB")
    image = image.resize((64, 64))
    image = np.array(image)
    input_image = np.expand_dims(image, axis=0)
    predict = model.predict(input_image)
    result = np.argmax(predict, axis=1)
    return result


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        f = request.files["file"]

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, "uploads", secure_filename(f.filename))
        f.save(file_path)
        value = get_result(file_path)
        result = get_class_name(value)
        return result
    return None


if __name__ == "__main__":
    app.run(debug=True)
