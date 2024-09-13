import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from torchvision import models, transforms
from torchvision.transforms import v2 as transforms
from PIL import Image


# Establecer el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import streamlit as st
from PIL import Image
import io
import numpy as np

st.title("Capture and Process Image for CNN")

# Capture the image from the user
image_data = st.camera_input("Take a picture")

if image_data is not None:
    # Open the image using PIL
    image = Image.open(image_data)
    
    # Convert image to PNG format and store it as bytes in memory
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Store the PNG byte array in session state
    st.session_state['captured_image_png'] = img_byte_arr
    st.success("PNG image stored temporarily for this session.")

    # Display the image from the PNG byte array
    st.image(image, caption='Captured Image')

# Processing the image for CNN model input
if 'captured_image_png' in st.session_state:
    # Load the PNG byte array as an image
    image_from_bytes = Image.open(io.BytesIO(st.session_state['captured_image_png']))
    
    # Optionally convert the image to the format/size expected by the model
    # For example, if your model expects 224x224 images:
    processed_image = image_from_bytes.resize((224, 224))
    
    # Convert the image to an array if needed by your model
    image_array = np.array(processed_image)
    
    # Placeholder for model prediction
    # result = model.predict(image_array.reshape(1, 224, 224, 3))  # Adjust as per your model's input requirements
    # st.write(f"Prediction result: {result}")

    # Display processed image ready for model input
    st.image(processed_image, caption='Processed Image Ready for CNN')
