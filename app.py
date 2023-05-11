from flask import Flask, render_template, request,  redirect
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import pandas as pd
from skimage import io
import cv2 as cv
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import VGG16

def save_webcam_image(img_path,iml_read):
    img = io.imread(iml_read)
    im = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    cv.imwrite(img_path, im)
    return True
def predict_image(model, img_path, preprocess_input_fn, decode_predictions_fn, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input_fn(x)
    preds = model.predict(x)
    predictions_df = pd.DataFrame(decode_predictions_fn(preds, top=10)[0])
    predictions_df.columns = ["Predicted Class", "Name", "Probability"]
    return predictions_df

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def index(): 
        tag_option = ''
        if request.method == "POST":
            tag_option = request.form.get('tag_name')
            iml_read = request.files["image"]
        if tag_option == 'Texted':
            model = tf.keras.models.load_model('my_text_model.h5')
            img_path = 'webcam_test_image.png'
            save_webcam_image(img_path,iml_read)
            image = cv.imread(img_path,cv.IMREAD_GRAYSCALE)       
            image = cv.resize(image, (28, 28))
            image = image.astype('float32')
            image = image.reshape(1, 28, 28, 1)
            image = 255-image
            image /= 255
            pred = model.predict(image.reshape(1,28,28,1), batch_size=1)
            return render_template("index.html", text_output=pred.argmax())
        elif tag_option =='Vision':
            name=pd.DataFrame()
            img_path = 'webcam_test_image.png'
            if save_webcam_image(img_path,iml_read) is False:img_path = "rocking_chair.jpg"
            model = VGG16()
            name=predict_image(model, img_path, vgg16.preprocess_input, vgg16.decode_predictions)
            return render_template("index.html",names=[name.to_html(classes='data', header="true")])
        return render_template('index.html',text_output=tag_option)
