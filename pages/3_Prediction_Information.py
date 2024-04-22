import streamlit as st
import keras
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



last_conv_layers = {'CatClassifier_v01':'conv2d_3','CatClassifier_v01':'conv2d_3', 'VGG16V_6':'conv2d_1'}


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
