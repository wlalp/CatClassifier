import streamlit as st
import pandas as pd
from PIL import Image
import tensorflow as tf
import keras
from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img, array_to_img


st.set_page_config("Prediction Page",":cat:")
st.markdown("<h1 style='text-align: center; color: white;'>Cat Breed Classifier</h1>", unsafe_allow_html=True)
model_name = "CatClassifier_v0"
model_file = "models/CatClassifier_v0.keras"
name_html = f"<p style='text-align: center; color: grey;font-family: Courier New'>Currently using [{model_name}]</p>"
name_holder = st.empty()
name_holder.markdown(name_html,unsafe_allow_html=True)

if 'model_name' not in st.session_state:
    st.session_state.model_name = model_name

file_col, breed_col,model_col = st.columns([0.4,.3,.3],gap="large")

def update_html(model_name):
    name_html = f"<p style='text-align: center; color: grey;font-family: Courier New'>Currently using [{model_name}]</p>"
    name_holder.markdown(name_html,unsafe_allow_html=True)

models = ("CatClassifier_v0","CatClassifier_v01","VGG16V_6")

with model_col:
    model_name = st.selectbox("Select a model",models,key="model_name")
    model_file = "models/"+ model_name + ".keras"
    update_html(model_name)

if model_name == "CatClassifier_v0":
    class_names = sorted(['Bombay', 'Bengal', 'Persian', 'Ragdoll', 'Sphynx', 'Russian Blue', 'Egyptian Mau',
               'Siamese', 'Birman', 'Abyssinian', 'British Shorthair', 'Maine Coon',])
else:
    class_names = sorted(['Bombay', 'Bengal', 'Persian', 'Ragdoll', 'Sphynx', 'Russian Blue', 'Egyptian Mau',
               'Siamese', 'Birman', 'Abyssinian', 'British Shorthair', 'Maine Coon','Domestic Shorthair'])
breed_col.header("Supported Breeds")




for cl in class_names:
    breed_col.write(cl)



model = keras.models.load_model(model_file, compile=False)
st.session_state.model = model

pred_data = pd.DataFrame(columns=["Breed","Confidence"])


uploaded_file = file_col.file_uploader("Upload your cat image here...", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = img.resize((256,256))
    st.session_state.img = img
    
    file_col.image(img,caption="Input Image", clamp=True, channels="RGB")

    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array,0)
    pred = model.predict(img_array)

    for p, cls in zip(pred[0],class_names):
        d = {"Breed":cls, "Confidence":(p * 100)}
        pred_data.loc[len(pred_data)] = d


    pred_data = pred_data.sort_values("Confidence",ascending=False)
    file_col.write("The model predicts...")
    file_col.dataframe(pred_data,hide_index=True,use_container_width=True)
    st.session_state.pred_df = pred_data
