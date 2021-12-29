### Run file through cmd
import streamlit as st
import io
import numpy as np
import tensorflow as tf
from PIL import Image
import efficientnet.tfkeras as efn

st.title("Plant disease classification")
st.write("Upload your plant leaf image and know if its healthy or not")

gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

#File upload
uploaded_file = st.file_uploader("Choose the leaf image", type = ['png', 'jpg'])

predictions_map = {0:"is healthy", 1:"multiple diseases", 2:"has rust", 3:"has scab"}

# load model
model = tf.keras.models.load_model("model.h5")

if uploaded_file is not None:

    # Read io bytes and convert into image
    image = Image.open(io.BytesIO(uploaded_file.read()))
    # display image
    st.image(image, use_column_width=True)
    # preprocess the image as per the model
    # resize the image, convert to numpy arrat and normalize
    resized_image = np.array(image.resize((512,512)))/255.
    image_batch = resized_image[np.newaxis, :, :, :]

    predictions_arr = model.predict(image_batch)
    predictions = np.argmax(predictions_arr)

    result = f"The plant {predictions_map[predictions]} with probability {predictions_arr[0][predictions]*100}%"
    if predictions == 0:
        st.success(result)
    else:
        st.error(result)