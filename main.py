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
model_file_id = '1-3_-XOrS7BUm-YsBgj5uzDzTNadx04qG'
model = load_model_from_google_drive(model_file_id, class_names).to(device)  # Here 'device' must be defined


# Example usage in Streamlit
if st.button('Load Model'):
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
from torchvision import models, transforms
import matplotlib.pyplot as plt
import pandas as pd
import io

# Configuraciones previas (diccionarios y transformaciones)
class_names = {...}  # Tu diccionario de clases
index_to_class = {v: k for k, v in class_names.items()}

transform = transforms.Compose([
    transforms.Lambda(lambda img: add_padding(img, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Tu modelo ya cargado, asumiendo que `model` es global y accesible aquí
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_and_show(pil_image):
    """Función para predecir y mostrar resultados directamente en Streamlit."""
    # Convertir PIL Image a tensor
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class_index = probabilities.argmax().item()
        predicted_class_name = index_to_class[predicted_class_index]
        predicted_probability = probabilities[predicted_class_index].item()

    # Mostrar la imagen en Streamlit
    st.image(pil_image, caption=f'Clase Predicha: {predicted_class_name} (Prob: {predicted_probability:.4f})')

    # Crear DataFrame para mostrar las probabilidades
    data = {
        'Clase': [index_to_class[i] for i in range(len(probabilities))],
        'Probabilidad': [f"{prob:.6f}" for prob in probabilities]
    }
    df = pd.DataFrame(data)
    st.dataframe(df)

# Captura de imagen desde la cámara
image = st.camera_input("Captura una imagen")
if image:
    st.success("Imagen capturada")

    # Convertir BytesIO a PIL Image
    pil_image = Image.open(image).convert("RGB")

    # Llamar a la función de predicción y mostrar
    predict_and_show(pil_image)
