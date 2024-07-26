import streamlit as st
import pandas as pd 
import numpy as np
import cv2
from PIL import Image

from sklearn.linear_model import LogisticRegression
import pickle



# Loading the pre-trained model
model = pickle.load(open(r"log.pkl","rb"))


# Title of the page
    
st.title("Dogs Breed Prediction")

#User input
image = st.file_uploader("Upload an image", type = ['jpg','jpeg','png'])

# Check if the mail is empty or not
if st.button('Submit'):
    if image:
        image = Image.open(image)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        res = cv2.resize(image,(150,150))
        flat = res.flatten()
        model.predict([flat])[0]
        st.image(image, caption='Uploaded Image', width=200)
#             st.image(r"C:\Users\Manisha\Downloads\spam image.png")
