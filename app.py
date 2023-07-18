import streamlit as st
import numpy as np
import pickle
import cv2 
import tensorflow as tf
from PIL import Image

# model = pickle.load(open('digitrecognizer.pkl', 'rb')) #using the saved model for predicting new handdrawn digit

# st.write(f"{type(model)}")
st.write(""" # Hello """)
st.markdown("<h1 style='text-align: center; color: white;'>Handwritten Digit Identification</h1>", unsafe_allow_html=True)
# st.write(""" ##### Upload an image and please wait for a few seconds """)

uploaded_file = st.file_uploader("Upload an image",type=("png"))

# if uploaded_file != None:
#     image = Image.open(uploaded_file)
#     img = np.array(image)

#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     resized = cv2.resize(gray,(28,28),interpolation = cv2.INTER_AREA)

#     newimg = tf.keras.utils.normalize(resized,axis=1)
#     newimg = np.array(newimg).reshape(-1,28,28,1)

#     predictions = model.predict(newimg) #predicting the hand drawn digit using model
  

#     c1, c2 = st.columns(2)

#     with c1:
#         st.image(uploaded_file,caption='The uploaded image of digit',width=200)
#     with c2:
#         st.write(f"### The digit identified is : {np.argmax(predictions)}")

# else:
#     st.write(""" ##### Upload an image and please wait for a few seconds """)
