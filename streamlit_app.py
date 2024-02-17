import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw
import tensorflow as tf
import keras
from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img, array_to_img

class_names = sorted(['bombay', 'bengal', 'persian', 'ragdoll', 'sphynx', 'russian_blue', 'egyptian_mau',
               'siamese', 'birman', 'abyssinian', 'british_shorthair', 'maine_coon'])

model = keras.models.load_model("CatClassifier_v0.keras", compile=False)

pred_data = pd.DataFrame(columns=["Breed","Confidence"])


uploaded_file = st.file_uploader("Upload image here...", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = img.resize((256,256))
    
    st.image(img,caption="Input Image", clamp=True, channels="RGB")

    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array,0)
    pred = model.predict(img_array)

    for p, cls in zip(pred[0],class_names):
        d = {"Breed":cls, "Confidence":(p * 100)}
        pred_data.loc[len(pred_data)] = d


    pred_data = pred_data.sort_values("Confidence",ascending=False)
    pred_data



    


