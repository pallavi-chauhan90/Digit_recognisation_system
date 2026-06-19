from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model("tf-cnn-model.h5", compile=False)


def predict_digit(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (28, 28))

    # Agar prediction galat aaye to uncomment karo
    # image = 255 - image

    image = image.astype("float32") / 255.0

    image = image.reshape(1, 28, 28, 1)

    prediction = model.predict(image, verbose=0)

    return np.argmax(prediction)


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None

    if request.method == "POST":

        file = request.files["image"]

        if file:

            filepath = os.path.join(
                UPLOAD_FOLDER,
                file.filename
            )

            file.save(filepath)

            prediction = predict_digit(filepath)

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)