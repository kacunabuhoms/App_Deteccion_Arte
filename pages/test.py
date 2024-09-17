import streamlit as st
import gspread
import io
import torch
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image
from torchvision.transforms import v2 as transforms

# Diccionario de clases
class_names = {'CLUSTER': 0, 'DANGLER': 1, 'KIT COPETE': 2, 'KIT DANG BOTADERO': 3,
               'MANTELETA': 4, 'MENU': 5, 'MP': 6, 'PC': 7, 'POSTER': 8,
               'PRECIADOR': 9, 'REFRICALCO': 10, 'STICKER': 11, 'STOPPER': 12, 'V UN': 13}

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
# TAKE THE IMAGE ----------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
st.title("Real-Time Image Classification with ResNet50")
# Capture an image from the camera
captured_image = st.camera_input("Take a picture")

# Convert the captured image to a PIL Image
if captured_image is not None:
    image = Image.open(io.BytesIO(captured_image.getvalue()))
    st.image(image, caption='Captured Image')





#--------------------------------------------------------------------------------------------------------
# LOAD MODEL --------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
# Cargar el modelo
def load_model(model_path):
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    num_classes = len(class_names)  # Ajustar al número de clases
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model_path = "Weights_ResNet50_Full layers_v3.pth"
model = load_model(model_path)




#--------------------------------------------------------------------------------------------------------
# PROCESS IMAGE -----------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
from torchvision import transforms

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

# Apply transformation
if captured_image is not None:
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model





#--------------------------------------------------------------------------------------------------------
# CLASSIFY ----------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
if captured_image is not None:
    with torch.no_grad():
        output = model(input_batch)
        # Assuming your model outputs a category index
        predicted_category = output.argmax(1).item()
        st.write(f'Predicted Category: {predicted_category}')

