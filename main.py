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

# Diccionario de clases
class_names = {'CLUSTER': 0, 'DANGLER': 1, 'KIT COPETE': 2, 'KIT DANG BOTADERO': 3,
               'MANTELETA': 4, 'MENU': 5, 'MP': 6, 'PC': 7, 'POSTER': 8,
               'PRECIADOR': 9, 'REFRICALCO': 10, 'STICKER': 11, 'STOPPER': 12, 'V UN': 13}

# Inversión del diccionario para obtener nombres por índice
index_to_class = {v: k for k, v in class_names.items()}

# Función para agregar padding y hacer la imagen cuadrada
def add_padding(img, size):
    old_width, old_height = img.size
    longest_edge = max(old_width, old_height)
    horizontal_padding = (longest_edge - old_width) / 2
    vertical_padding = (longest_edge - old_height) / 2
    padded_img = Image.new("RGB", (longest_edge, longest_edge), color=(0, 0, 0))
    padded_img.paste(img, (int(horizontal_padding), int(vertical_padding)))
    return padded_img.resize((size, size), Image.LANCZOS)

# Transformaciones
transform = transforms.Compose([
    transforms.Lambda(lambda img: add_padding(img, 256)),
    transforms.CenterCrop(224),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Supongamos que el modelo ya está cargado en 'model'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Función para predecir y mostrar la imagen y la tabla de probabilidades lado a lado
def predict_and_show(pil_image):
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class_index = probabilities.argmax().item()
        predicted_class_name = index_to_class[predicted_class_index]
        predicted_probability = probabilities[predicted_class_index].item()

    # Configurar el plot para mostrar imagen y tabla lado a lado
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(pil_image)
    ax[0].set_title(f'Clase Predicha: {predicted_class_name} (Prob: {predicted_probability:.4f})')
    ax[0].axis('off')

    data = {'Clase': [index_to_class[i] for i in range(len(probabilities))],
            'Probabilidad': [f"{prob:.6f}" for prob in probabilities]}
    df = pd.DataFrame(data)
    ax[1].axis('off')
    table = ax[1].table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.show()

# Captura de imagen desde la cámara
image = st.camera_input("Captura una imagen")
if image:
    st.success("Imagen capturada")

    # Convertir BytesIO a PIL Image y luego a PNG
    pil_image = Image.open(image).convert("RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    pil_image_png = Image.open(buffer)

    # Llamar a la función de predicción y mostrar
    predict_and_show(pil_image_png)

