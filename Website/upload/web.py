from flask import Flask, render_template, request, redirect

import base64
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from PIL import Image

# Load the model
import numpy as np
import os
# import cv2
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
# model = load_model("static/model/40_epochs.h5")
# model_color = load_model("static/model/color_epochs.h5")

model = load_model("static/model/wiki_model.h5")
model_color = load_model("static/model/wiki_model2.h5")

encoded = ""

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

    # if (i.shape)[0] < (i.shape)[1]:
    #     crop_size = i.shape[0]
    # else:
    #     crop_size = i.shape[1]

    # im_new = crop_center(i,crop_size,crop_size)
    
    IMG_SIZE = 100

    # Black and White
    img_gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    new_array = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    # plt.imshow(new_array, cmap = "gray")


    # Color
    new_array_c = cv2.resize(i, (IMG_SIZE, IMG_SIZE))
    # plt.imshow(new_array_c)

    X = np.array(new_array).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    X_c = np.array(new_array_c).reshape(-1,IMG_SIZE,IMG_SIZE,1)

    X = X/255
    X_c = X_c/255

    prediction = model.predict(X)[0][0]
    prediction = abs(1-prediction)
    male_prediction = round(((1-prediction)*100),2)
    female_prediciton = round(prediction*100,2)

    higher = []

    if (prediction < .5):
        print(f"Male with {male_prediction}% certainty")
        higher = ["Male",male_prediction]
    else:
        print(f"Female with {female_prediciton}% certainty")
        higher = ["Female",female_prediciton]
        

    prediction_c = model_color.predict(X_c)[0][0]
    prediction_c = abs(1-prediction_c)

    male_prediction_c = round(((1-prediction_c)*100),2)
    female_prediction_c = round(prediction_c*100,2)

    if (prediction_c < .5):
        print(f"Male with {male_prediction_c}% certainty")
        higher_c = ["Male",male_prediction_c]

    else:
        print(f"Female with {female_prediction_c}% certainty")
        higher_c = ["Female",female_prediction_c]

    if higher[1] > higher_c[1]:
        highest = higher.copy()
    else:
        highest = higher_c.copy()

    
    return render_template("index.html",percent = highest[1], gender = highest[0], encoded=encoded)

def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[starty:starty+cropy, startx:startx+cropx, :]


if __name__ == "__main__":
    app.run(debug=True)