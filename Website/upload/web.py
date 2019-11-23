from flask import Flask, render_template, request, redirect

import base64
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the model
import numpy as np
import os

# import cv2
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

model_1 = load_model("static/model/40_epochs.h5")
model_2 = load_model("static/model/color_epochs.h5")
model_3 = load_model("static/model/wiki_model.h5")
model_4 = load_model("static/model/wiki_model2.h5")
model_5 = load_model("static/model/Everything.h5")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("homepage.html")

@app.route("/camera")
def camera():
    return render_template("index.html")

@app.route("/engine")
def engine():
    return render_template("engine.html")

@app.route('/data', methods=['POST'])
def handle_data():
    encoded = request.form['datauri']
    clean = encoded.split(",")[1]
    
    # Convert Uri to jpg
    i = base64.b64decode (str(clean))
    i = io.BytesIO(i)
    i = mpimg.imread(i, format='JPG')

    
    IMG_SIZE = 100

    # Black and White
    img_gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    new_array = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))


    # Color
    new_array_c = cv2.resize(i, (IMG_SIZE, IMG_SIZE))


    X = np.array(new_array).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    X_c = np.array(new_array_c).reshape(-1,IMG_SIZE,IMG_SIZE,3)

    X = X/255
    X_c = X_c/255

    final = round(model_1.predict(X)[0][0]) \
        +round(model_2.predict(X_c)[0][0]) \
        + round(abs(1-model_3.predict(X)[0][0])) \
        + round(abs(1-model_4.predict(X)[0][0])) \
        + round(abs(1-model_5.predict(X)[0][0]))

    print (f"model 1:{model_1.predict(X)[0][0]}")
    print (f"model 2:{model_2.predict(X_c)[0][0]}")
    print (f"model 3:{abs(1-model_3.predict(X)[0][0])}")
    print (f"model 4:{abs(1-model_4.predict(X)[0][0])}")
    print (f"model 5:{abs(1-model_5.predict(X)[0][0])}")

    final = final / 5
    if final <0.5 :
        # print ("male")
        gender = "MALE"
    else: 
        # print("female")
        gender = "FEMALE"
    
    return render_template("index.html",gender = gender, encoded=encoded)


if __name__ == "__main__":
    app.run(debug=True)