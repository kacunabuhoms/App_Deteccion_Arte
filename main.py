import streamlit as st
import gspread
import pandas as pd
from datetime import datetime
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image
import pytz
import torch
from torchvision import models, transforms



tz = pytz.timezone('America/Monterrey')

# Configuración de gspread para conectar con Google Sheets
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=SCOPES
)


# Construye el servicio de la API de Google Drive
service = build('drive', 'v3', credentials=credentials)

# ID del archivo en Google Drive que deseas descargar
file_id = '1xIIzJsNCfuTpxAgXehy2r7QVEIsnl7Ks'
request = service.files().get_media(fileId=file_id)
fh = io.BytesIO()
downloader = MediaIoBaseDownload(fh, request)
done = False
while done is False:
    status, done = downloader.next_chunk()

# Utiliza PIL para abrir la imagen desde el stream de bytes
fh.seek(0)
image = Image.open(fh)

# Mostrar la imagen en Streamlit
st.logo(image)

#------------------------

# Assuming 'service' setup is done outside this snippet based on your previous setup

@st.cache(allow_output_mutation=True)
def load_model_from_google_drive(file_id):
    """Load model from Google Drive into PyTorch model."""
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    
    # Load the model state dict from the byte stream
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    # Adjust the line below to match the number of classes you have
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(fh, map_location=torch.device('cpu')))
    model.eval()
    return model

# Your specific model file ID on Google Drive
model_file_id = '1-3_-XOrS7BUm-YsBgj5uzDzTNadx04qG'
model = load_model_from_google_drive(model_file_id).to(device)

# Example usage in Streamlit
if st.button('Load Model'):
    st.write("Model loaded successfully!")
    # You can add more functionality here to utilize the model

