import streamlit as st
import gspread
import io
import torch
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image
from torchvision.transforms import v2 as transforms



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





#--------------------------------------------------------------------------------------------------------
# TAKING THE PHOTO --------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

# Function: Add padding to the image
def add_padding(img, size):
    old_width, old_height = img.size
    longest_edge = max(old_width, old_height)
    horizontal_padding = (longest_edge - old_width) / 2
    vertical_padding = (longest_edge - old_height) / 2
    padded_img = Image.new("RGB", (longest_edge, longest_edge), color=(0, 0, 0))
    padded_img.paste(img, (int(horizontal_padding), int(vertical_padding)))
    return padded_img.resize((size, size), Image.LANCZOS)

# Define image transformations
transform = transforms.Compose([
    transforms.Lambda(lambda img: add_padding(img, 256)),
    transforms.CenterCrop(224),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.write("Take a photo to classify:")
captured_image = st.camera_input("Click the button to capture an image:")

if captured_image:
    # Convert the captured image to a PIL Image
    image_bytes = captured_image.getvalue()
    pil_image = Image.open(io.BytesIO(image_bytes))

    # Display the captured image
    st.image(pil_image, caption='Captured Image', use_column_width=True)

    # Transform the image for the model
    input_tensor = transform(pil_image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    if st.button('Classify Image'):
        # Move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)
            # Assuming the model returns a tensor of category probabilities
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            # Display the top category
            top_prob, top_catid = torch.max(probabilities, dim=0)
            st.write(f'Predicted Category: {top_catid}, Probability: {top_prob.item()}')
