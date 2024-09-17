import streamlit as st
import gspread
import io
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

FOLDER_ID = "1e89Hs6yvZWZ-4Rz0cmIc07GV3mYrBgWS"  # Your Google Drive Folder ID

# Option to load and display the selected file
if st.button('Load File'):
    # Assuming these are text files or some readable format
    file_id = next(file['id'] for file in files if file['name'] == selected_file)
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    try:
        while done is False:
            status, done = downloader.next_chunk()
        fh.seek(0)
        content = fh.read().decode()
        st.text_area("File Content", content, height=300)
    except Exception as e:
        st.error("Failed to read file: " + str(e))