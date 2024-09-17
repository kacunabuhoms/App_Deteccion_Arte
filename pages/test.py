import streamlit as st
from PIL import Image
import io
import torch
from torchvision import models, transforms

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
# PAGE CONFIGURATION ------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

# Dictionary of classes
class_names = {
    'CLUSTER': 0, 'DANGLER': 1, 'KIT COPETE': 2, 'KIT DANG BOTADERO': 3,
    'MANTELETA': 4, 'MENU': 5, 'MP': 6, 'PC': 7, 'POSTER': 8,
    'PRECIADOR': 9, 'REFRICALCO': 10, 'STICKER': 11, 'STOPPER': 12, 'V UN': 13
}

# Load the model
def load_model(model_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# Function to add padding and make the image square
def add_padding(img, size):
    old_width, old_height = img.size
    longest_edge = max(old_width, old_height)
    horizontal_padding = (longest_edge - old_width) / 2
    vertical_padding = (longest_edge - old_height) / 2
    padded_img = Image.new("RGB", (longest_edge, longest_edge), color=(0, 0, 0))
    padded_img.paste(img, (int(horizontal_padding), int(vertical_padding)))
    return padded_img.resize((size, size), Image.LANCZOS)

# Transformations
transform = transforms.Compose([
    transforms.Lambda(lambda img: add_padding(img, 256)),
    transforms.CenterCrop(224),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image, model, device):
    img_t = transform(image)
    img_t = img_t.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_t)
        _, predicted = torch.max(output, 1)
    class_name = class_names[predicted.item()]
    return class_name

# Streamlit app setup
st.title('Real-Time Image Classification with ResNet50')
camera_image = st.camera_input("Take a picture")

if camera_image is not None:
    image = Image.open(io.BytesIO(camera_image.getvalue()))
    st.image(image, caption='Captured Image', use_column_width=True)

    st.write("Loading model and classifying image...")
    # Assume the model file is already handled by your API connection setup
    model, device = load_model('/content/drive/MyDrive/Proyectos/Deteccion Arte/CNN/ResNet50/Modelo/Full_ResNet50_Ful layers_v3.pth')  # Ensure you adjust this path
    
    label = predict_image(image, model, device)
    st.write(f'Prediction: {label}')
