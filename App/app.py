#importing all the required libraries
import os.path
import streamlit as st 
import tensorflow as tf
import cv2
import keras
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
import numpy as np
from PIL import Image ,ImageOps

#Path where you want to save the uploaded images
save_path = "C:/Users/Mihir/Desktop/all folders/2nd Year/4th Sem/MiniProject/App/Static"

#on loading a streamlit app we get a warning, this line prevents us from getting that warning
st.set_option('deprecation.showfileUploaderEncoding',False) 

#this line prevent us from loading the model again and again and will help in storing the model in cache once it has been loaded
@st.cache(allow_output_mutation=True) 

#loading our model
def load_model(): 
  model = tf.keras.models.load_model('my_model.h5')
  return model

model = load_model()


#defining the header or title of the page that the user will be seeing. We also make a side bar for the web app
st.markdown("<h1 style='text-align: center; color: Black;'>Breast Cancer Detection Using CNN</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: Black;'>All you have to do is Upload the MRI scan/Histopahology images and the model will do the rest!</h3>", unsafe_allow_html=True)
st.sidebar.header("What is this Project about?")
st.sidebar.text("It is a Deep learning solution to detection of Breast Cancer using CNN model")
st.sidebar.header("What does it do?")
st.sidebar.text("The user can upload their MRI scan and the model will try to predict whether or not the user has Brain Tumor or not.")
st.sidebar.header("What tools where used to make this?")
st.sidebar.text("The Model was made using a dataset from Kaggle along with using Kaggle notebooks to train the model. We made use of Tensorflow, Keras as well as some other Python Libraries to make this complete project. To deply it on web, we used ngrok and Streamlit!")


#accepting the image input from the user
file=st.file_uploader("Please upload your MRI Scan",type = ["jpg","png"])

#initial condition when no image has been uploaded by the user
if file is None: 
  st.markdown("<h5 style='text-align: center; color: Black;'>Please Upload a File</h5>", unsafe_allow_html=True)
#condition to give the result once the user has input the image
else:  
  image1 = Image.open(file)
  print(image1)
  st.image(image1,use_column_width = True)
  #saving the file to local machine
  with open(os.path.join(save_path,file.name),"wb") as f: 
    f.write(file.getbuffer())         
  st.success("File Succuessfully Saved")
  #loading the image into data and converting it into an array
  data = image.load_img(os.path.join(save_path,file.name), target_size=(48, 48, 3))
  data = np.expand_dims(data, axis=0)
  data = data * 1.0 / 255
  #Predict the result
  predicted = model.predict(data)
  result = predicted 
  indices = {0: 'Benign', 1: 'Malignant'}
  predicted_class = np.asscalar(np.argmax(result, axis=1))
  print(predicted_class)
  string = "The patient most likely has: "+ indices[predicted_class]
  #Printing the result
  if predicted_class == 0:
    st.success(string)
  else:
    st.error(string)