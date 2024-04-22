import streamlit as st
import keras
from keras import Model
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array



last_conv_layers = {'CatClassifier_v0':'conv2d_3','CatClassifier_v01':'conv2d_3', 'VGG16V_6':'conv2d_1'}
breed_links = ['https://www.purina.com/cats/cat-breeds/abyssinian-cat',
               'https://www.purina.com/cats/cat-breeds/bengal-cat',
               'https://www.purina.com/cats/cat-breeds/birman-cat',
               'https://www.purina.com/cats/cat-breeds/bombay',
               'https://www.purina.com/cats/cat-breeds/british-shorthair',
               'https://www.purina.com/cats/cat-breeds/american-shorthair',
               'https://www.purina.com/cats/cat-breeds/egyptian-mau',
               'https://www.purina.com/cats/cat-breeds/maine-coon',
               'https://www.purina.com/cats/cat-breeds/persian',
               'https://www.purina.com/cats/cat-breeds/ragdoll',
               'https://www.purina.com/cats/cat-breeds/russian-blue',
               'https://www.purina.com/cats/cat-breeds/siamese',
               'https://www.purina.com/cats/cat-breeds/sphynx'
               ]
def get_conv_layers(model):
    l_index = 0
    conv_layers = []
    for l in model.layers:
        if type(l) == keras.layers.Conv2D:
            conv_layers.append(l_index)
        l_index += 1
    return conv_layers

def show_feature_maps(model,index):
    show_model = Model(inputs = model.inputs, outputs = model.layers[index].output)
    feature_maps = show_model.predict(img_array)
    
    square = 4
    ix = 1
    for _ in range(square):
        for _ in range(square):
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])

            plt.imshow(feature_maps[0, :, :, ix-1])
            ix += 1

    st.pyplot()


try:
    pred_df = st.session_state.pred_df
    pred_row = pred_df['Confidence'].idxmax()
    pred_breed = pred_df['Breed'][pred_row]
    st.markdown(f"<h1 style='text-align: center;'>The model predicted... {pred_breed}!</h1>", unsafe_allow_html=True)
    st.dataframe(pred_df,hide_index=True)
except:
    st.write("No prediction found! Please input an image on the main page to see your results here.")


if 'model' in st.session_state and 'img' in st.session_state and 'model_name' in st.session_state:
    st.write("Let's take a look at what the model saw!")

    model = st.session_state.model
    model_name = st.session_state.model_name
    img = st.session_state.img
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array,0)

    class_names = sorted(['Bombay', 'Bengal', 'Persian', 'Ragdoll', 'Sphynx', 'Russian Blue', 'Egyptian Mau',
                'Siamese', 'Birman', 'Abyssinian', 'British Shorthair', 'Maine Coon','Domestic Shorthair'])
    
    link_index = 0
    link_dict = dict()
    for c in class_names:
        link_dict[c] = breed_links[link_index]
        link_index += 1

    layer_indices = get_conv_layers(model)
    for li in layer_indices:
        show_feature_maps(model,li)


    st.write(f"Learn more about the {pred_breed} here: ",link_dict[pred_breed])
    
    
