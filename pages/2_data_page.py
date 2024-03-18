import streamlit as st
from PIL import Image
import requests

st.markdown("<h1 style='text-align: center; color: white;'>Data Information</h1>", unsafe_allow_html=True)
st.markdown("<hp style='text-align: center; color: white;'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The data collected is of the Domestic Shorthair breed. A breed that is missing from the Oxford Pet dataset. The breed's exclusion from the dataset is a mystery, but it does not mean the breed should remain excluded. However, with the data collected it can be easily added along side that dataset to extend its scope. The most common domesticated cat in the United States,and much of the world, is the Domestic Shorthair. So, for the purpose of classifying user's cats its important to account for what will most likely be a large majority of the predicted breeds.</p>", unsafe_allow_html=True)

st.markdown("<a href='https://github.com/wlalp/Domestic-Shorthairs' style='text-align: center; color: white;'>https://github.com/wlalp/Domestic-Shorthairs</a>",unsafe_allow_html=True)