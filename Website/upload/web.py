# from flask import Flask, render_template, request, redirect
# from PIL import Image

# # Load the model
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# # import cv2
# import cv2
# # import os
# import tensorflow as tf
# from tensorflow.keras.models import load_model

# model = load_model("static/model/40_epochs.h5")
# model_color = load_model("static/model/color_epochs.h5")


# import base64
# import io
# from matplotlib import pyplot as plt
# import matplotlib.image as mpimg

# app = Flask(__name__)

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route('/data', methods=['POST'])
# def handle_data():
#     encoded = request.form['datauri']
    
#     # Convert Uri to jpg
#     i = base64.b64decode (encoded)
#     i = io.BytesIO(i)
#     i = mpimg.imread(i, format='JPG')

#     # plt.imshow(i, interpolation='nearest')
#     # plt.show()  
    
#     # Crop image
#     im_new = crop_center(i)

#     # Run model
#     path = i
#     IMG_SIZE = 100  

#     # Black and White
#     img_array = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
#     new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#     # plt.imshow(new_array, cmap = "gray")


#     # Color
#     img_array_c = cv2.imread(i)
#     RGB_img = cv2.cvtColor(img_array_c, cv2.COLOR_BGR2RGB)
#     new_array_c = cv2.resize(RGB_img, (IMG_SIZE, IMG_SIZE))
#     plt.imshow(RGB_img)


#     return redirect("/")

# def crop_center(pil_img):
#     minimum = min(pil_img.size)
#     img_width, img_height = pil_img.size
#     return pil_img.crop(((img_width - minimum) // 2,
#                          (img_height - minimum) // 2,
#                          (img_width + minimum) // 2,
#                          (img_height + minimum) // 2))

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request, redirect

import base64
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# Load the model
import numpy as np
import os
# import cv2
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
model = load_model("static/model/40_epochs.h5")
model_color = load_model("static/model/color_epochs.h5")


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/data', methods=['POST'])
def handle_data():
    encoded = request.form['datauri']
    encoded = encoded.split(",")[1]
    
    # Convert Uri to jpg
    i = base64.b64decode (str(encoded))
    i = io.BytesIO(i)
    i = mpimg.imread(i, format='JPG')

    if (i.shape)[0] < (i.shape)[1]:
        crop_size = i.shape[0]
    else:
        crop_size = i.shape[1]

    im_new = crop_center(i,crop_size,crop_size)
    
    IMG_SIZE = 100

    # Black and White
    img_gray = cv2.cvtColor(im_new, cv2.COLOR_BGR2GRAY)
    new_array = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    plt.imshow(new_array, cmap = "gray")


    # Color
    new_array_c = cv2.resize(im_new, (IMG_SIZE, IMG_SIZE))
    plt.imshow(new_array_c)

    X = np.array(new_array).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    X_c = np.array(new_array_c).reshape(-1,IMG_SIZE,IMG_SIZE,3)

    X = X/255
    X_c = X_c/255

    prediciton = model.predict(X)[0][0]
    male_prediction = round(((1-prediciton)*100),2)
    female_prediciton = round(prediciton*100,2)

    if (prediciton < .5):
        print(f"Male with {male_prediction}% certainty")
    #     print(f"{round(male_prediction,2)}")
    else:
            print(f"Female with {female_prediciton}% certainty")

    prediciton_c = model_color.predict(X_c)[0][0]
    male_prediction_c = round(((1-prediciton_c)*100),2)
    female_prediciton_c = round(prediciton_c*100,2)

    if (prediciton_c < .5):
        print(f"Male with {male_prediction_c}% certainty")
    #     print(f"{round(male_prediction,2)}")
    else:
            print(f"Female with {female_prediciton_c}% certainty")


    return redirect("/")

def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[starty:starty+cropy, startx:startx+cropx, :]


if __name__ == "__main__":
    app.run(debug=True)