#To run the application:
#    streamlit run main_app.py
#Library imports
import tensorflow as tf
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model


#Loading the Model
model = load_model('falcon_types.h5')

#Name of Classes
CLASS_NAMES = ['Gyr', 'Saker', 'Peregrine']

#Setting Title of App
st.title("Falcon Type Prediction")
st.markdown("Upload an image of the Falcon")

#Uploading the dog image
falcon_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict')
#On predict button click
if submit:


    if falcon_image is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(falcon_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (280,280))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,280,280,3)
        #Make Prediction
        predictions = model.predict(opencv_image)
        score = tf.nn.softmax(predictions[0])


        st.title(str("This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(CLASS_NAMES[np.argmax(score)], 100 * np.max(score))))

        # Displaying the image
        st.image(opencv_image, channels="BGR")
        #st.write(opencv_image.shape)