import streamlit as st
import gspread
import io
import torch
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image

#--------------------------------------------------------------------------------------------------------
# PAGE CONFIGURATION ------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

st.title("Demo de detección de arte")

# Configuración de gspread para conectar con Google Sheets
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=SCOPES
)
# Usar las credenciales para autenticarse con Google Sheets
gc = gspread.authorize(credentials)
sh = gc.open_by_key('1D8V_C7tUZ4qiNlZiba96a3jMDPlQ-tMFPTfuuJ3dEAw')

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

#--------------------------------------------------------------------------------------------------------
# LOADING THE MODEL -------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
FOLDER_ID = "1e89Hs6yvZWZ-4Rz0cmIc07GV3mYrBgWS"  # Your Google Drive Folder ID

st.text("Modelos disponibles:")

# Function to list files
def list_files(service, folder_id):
    results = service.files().list(
        q=f"'{folder_id}' in parents",
        pageSize=10,
        fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])
    return items

# Streamlit interface
st.text("Modelos disponibles:")
files = list_files(service, FOLDER_ID)
file_names = [file['name'] for file in files]
selected_file = st.selectbox('Select a file:', file_names)

# Function to load .pth file into PyTorch model
def load_model(file_id):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    fh.seek(0)
    # Load the model state
    model = torch.load(fh, map_location=torch.device('cpu'))  # Adjust as necessary
    return model

if st.button('Load Model'):
    file_id = next(file['id'] for file in files if file['name'] == selected_file)
    model = load_model(file_id)
    st.write("Model loaded successfully!")