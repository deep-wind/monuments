# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 23:20:19 2022

@author: PRAMILA
"""
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf 


def load_image(image_file):
	img = Image.open(image_file)
	if(image_file.type=="image/png"):
	   picture = img.save("monuments.png") 
	if(image_file.type=="image/jpg"):
	   picture = img.save("monuments.jpg") 
	return img



st.markdown("<h1 style ='color:black; text_align:center;font-family:times new roman;font-size:20pt; font-weight: bold;'>Monuments Identification from Satellite Images</h1>", unsafe_allow_html=True)
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

#if image_file is not None:
	 # To See details
file_details = {"filename":image_file.name, "filetype":image_file.type,
                  "filesize":image_file.size}
st.write(file_details)

# To View Uploaded Image
st.image(load_image(image_file),width=200)

#image_file=load_image(image_file)

st.write(image_file.type)
categories = ['ajantacaves','amaravathistupa','charminar','gatewayofindia','golgumbaz','indiagate','kanchmahal','konraksuntemple','qutubminar','tajmahal','vivekanandharock']

model = tf.keras.models.load_model('model.h5')
#path = r"C:\Users\PRAMILA\.spyder-py3\project\monuments\ajantacaves\ajantacaves3.png"
if(image_file.type=="image/png"):
   img = image.load_img("monuments.png", target_size=(224,224))
if(image_file.type=="image/jpg"):
    img = image.load_img("monuments.jpg", target_size=(224,224))
#st.write(plt.imshow(img))
x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

imagea = np.vstack([x])
classes = model.predict(imagea)
print(classes)
type(classes)
max_index_col = np.argmax(classes, axis=1)

print(max_index_col)
st.success(categories[int(max_index_col)])
