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
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#Load dataset
data_dir = pathlib.Path("monuments/monuments/")
image_count = len(list(data_dir.glob('*/*.png')))
print(image_count)



#Read monuments images from disk into numpy array using opencv
monuments_images_dict = {
    'ajantacaves': list(data_dir.glob('ajantacaves/*')),
    'amaravathistupa': list(data_dir.glob('amaravathistupa/*')),
    'charminar': list(data_dir.glob('charminar/*')),
    'gatewayofindia(mumbai)': list(data_dir.glob('gatewayofindia(mumbai)/*')),
    'golgumbaz': list(data_dir.glob('golgumbaz/*')),
    'indiagate': list(data_dir.glob('indiagate/*')),
    'kanchmahal': list(data_dir.glob('kanchmahal/*')),
    'konraksuntemple': list(data_dir.glob('konraksuntemple/*')),
    'qutubminar':list(data_dir.glob('qutubminar/*')),
    'tajmahal': list(data_dir.glob('tajmahal/*')),
    'VivekanandaRock': list(data_dir.glob('VivekanandaRock/*'))
}

monuments_labels_dict = {
    'ajantacaves': 0,
    'amaravathistupa': 1,
    'charminar': 2,
    'gatewayofindia(mumbai)': 3,
    'golgumbaz': 4,
    'indiagate':5,
    'kanchmahal':6,
    'konraksuntemple':7,
    'qutubminar':8,
    'tajmahal': 9,
    'VivekanandaRock':10
}

monuments_dict =['ajantacaves','amaravathistupa','charminar','gatewayofindia(mumbai)','golgumbaz','indiagate','kanchmahal','konraksuntemple','qutubminar','tajmahal','vivekanandharock']

#Train test split
X, y = [], []

for monument_name, images in monuments_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img,(180,180))
        X.append(resized_img)
        y.append(monuments_labels_dict[monument_name])

X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

num_classes = 11

model = Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
model.fit(X_train_scaled, y_train, epochs=50)             

model.save('model.h5')



def load_image(image_file):
	img = Image.open(image_file)
	if(image_file.type=="image/png"):
	   picture = img.save("monuments.png") 
	if(image_file.type=="image/jpg"):
	   picture = img.save("monuments.jpg") 
	return img



st.markdown("<h1 style ='color:black; text_align:center;font-family:times new roman;font-size:20pt; font-weight: bold;'>Monuments Identification from Satellite Images</h1>", unsafe_allow_html=True)
image_file = st.file_uploader("Upload Images", type=["png","jpg"])


if image_file is not None:
	 # To See details
    file_details = {"filename":image_file.name, "filetype":image_file.type,
                      "filesize":image_file.size}
    st.write(file_details)
    
    # To View Uploaded Image
    st.image(load_image(image_file),width=200)
    
    #image_file=load_image(image_file)
    
  #  st.write(image_file.type)
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
