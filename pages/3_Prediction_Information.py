import streamlit as st
import keras
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



last_conv_layers = {'CatClassifier_v01':'conv2d_3','CatClassifier_v01':'conv2d_3', 'VGG16V_6':''}


try:
    pred_df = st.session_state.pred_df
    pred_row = pred_df['Confidence'].idxmax()
    pred_breed = pred_df['Breed'][pred_row]
    st.markdown(f"<h1 style='text-align: center;'>The model predicted... {pred_breed}!</h1>", unsafe_allow_html=True)
    st.dataframe(pred_df,hide_index=True)
except:
    st.write("No prediction found! Please input an image on the main page to see your results here.")

#Reference: https://keras.io/examples/vision/grad_cam/
def heatmap(img,model,last_conv_layer):
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img,axis=0)

    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


if 'model' in st.session_state and 'img' in st.session_state:
    st.write("Let's take a look at what the model saw!")

    model = st.session_state.model
    model_name = st.session_state.model_name
    img = st.session_state.img
    heat = heatmap(img,model, last_conv_layers[model_name])

    st.pyplot(heat)