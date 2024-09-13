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


#------------------------
# Credenciales de Google API
#------------------------

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
# Carga del modelo
#------------------------

class_names = {'CLUSTER': 0, 'DANGLER': 1, 'KIT COPETE': 2, 'KIT DANG BOTADERO': 3,
               'MANTELETA': 4, 'MENU': 5, 'MP': 6, 'PC': 7, 'POSTER': 8,
               'PRECIADOR': 9, 'REFRICALCO': 10, 'STICKER': 11, 'STOPPER': 12, 'V UN': 13}

# Define the device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache(allow_output_mutation=True)
def load_model_from_google_drive(file_id, class_names):
    # Ensure service setup for Google Drive API is here or passed appropriately
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    
    # Load and set up the model
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(fh, map_location=device))  # Ensure 'device' is known here
    model.eval()
    return model

# Make sure to define 'class_names' appropriately before this point

# Example usage in Streamlit
if st.button('Load Model'):
    model_file_id = '1-3_-XOrS7BUm-YsBgj5uzDzTNadx04qG'
    model = load_model_from_google_drive(model_file_id, class_names).to(device)  # Here 'device' must be defined
    st.write("Model loaded successfully!")
    # You can add more functionality here to utilize the model



#------------------------
# Captura de imagen
#------------------------

st.title("Demo de detección de arte")





#------------------------
# 
#------------------------
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import io

model.eval()

# Define the transformation
def add_padding(img, size):
    old_width, old_height = img.size
    longest_edge = max(old_width, old_height)
    horizontal_padding = (longest_edge - old_width) / 2
    vertical_padding = (longest_edge - old_height) / 2
    padded_img = Image.new("RGB", (longest_edge, longest_edge), color=(0, 0, 0))
    padded_img.paste(img, (int(horizontal_padding), int(vertical_padding)))
    return padded_img.resize((size, size), Image.LANCZOS)

transform = transforms.Compose([
    transforms.Lambda(lambda img: add_padding(img, 256)),
    transforms.CenterCrop(224),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title('Image Classification App')

# Use st.camera_input to capture an image
captured_image = st.camera_input("Take a picture")

if captured_image is not None:
    st.image(captured_image, caption='Captured Image', use_column_width=True)
    
    # Button to trigger prediction
    if st.button('Classify Image'):
        # Convert the image to PNG format using PIL
        image = Image.open(captured_image)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Reload the image from the byte array
        image = Image.open(io.BytesIO(img_byte_arr))

        st.write("Classifying...")

        # Transform the image to be compatible with your model
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

        # Move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)

        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Show the top 5 categories
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        for i in range(top5_prob.size(0)):
            st.write(f"{top5_prob[i].item():.6f} : {class_names[top5_catid[i]]}")  # 'categories' should be the list of class names

