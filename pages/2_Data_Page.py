import streamlit as st
from PIL import Image
import requests

st.markdown("<h1 style='text-align: center;'>Data Information</h1>", unsafe_allow_html=True)
example_file = "example_images.png"
img = Image.open(example_file)
st.image(img,caption="Example images from the dataset")

st.markdown("<hp style='text-align: center;'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The data collected, using PetFinder's API is of the Domestic Shorthair breed. A breed that is missing from the Oxford Pet dataset. The breed's exclusion from the dataset is a mystery, but it does not mean the breed should remain excluded. With the data collected, domestic shorthair image data can be easily added alongside the Oxford-IIIT Pet dataset to extend its scope. The most common domesticated cat in the United States, and much of the world, is the Domestic Shorthair. So, for the purpose of classifying users' cats its important to account for what will most likely be a large majority of the input cats' breeds. A link to our dataset as well as is provided below, alongside a link to the Oxford-IIIT Pet dataset, and the landing page for PetFinder's API.</p>", unsafe_allow_html=True)
st.markdown("<h3>Relevant Links<h3>",unsafe_allow_html=True)
st.markdown("<p> Our dataset: <a href='https://github.com/wlalp/Domestic-Shorthairs'>https://github.com/wlalp/Domestic-Shorthairs</a></p>",
            unsafe_allow_html=True)
st.markdown("<p> Oxford-IIIT Pet dataset: <a href='https://www.robots.ox.ac.uk/~vgg/data/pets/'>https://www.robots.ox.ac.uk/~vgg/data/pets/</a></p>",
            unsafe_allow_html=True)
st.markdown("<p> Petfinder API: <a href='https://www.petfinder.com/developers/'>https://www.petfinder.com/developers/</a></p>",
            unsafe_allow_html=True)
