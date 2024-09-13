import streamlit as st
import torch
from torchvision import models, transforms
from torchvision.transforms import v2 as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import pandas as pd
import tempfile

st.title("Detección de Arte")

image = st.camera_input("Capturar imagen")

if image:
    st.success("Imagen capturada")

    # Convert BytesIO object to PIL Image
    pil_image = Image.open(image)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        temp_file_path = temp_file.name
        pil_image.save(temp_file_path, format="PNG")

    # Display the image
    st.image(pil_image)

    # Confirm the temporary file path
    st.write(f"Imagen guardada temporalmente en {temp_file_path}")