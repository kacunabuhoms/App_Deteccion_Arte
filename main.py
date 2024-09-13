import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import io

# Initialize the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names and index to class mapping
class_names = {'CLUSTER': 0, 'DANGLER': 1, 'KIT COPETE': 2, 'KIT DANG BOTADERO': 3,
               'MANTELETA': 4, 'MENU': 5, 'MP': 6, 'PC': 7, 'POSTER': 8,
               'PRECIADOR': 9, 'REFRICALCO': 10, 'STICKER': 11, 'STOPPER': 12, 'V UN': 13}
index_to_class = {v: k for k, v in class_names.items()}

# Define the transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Google Drive API Setup
SCOPES = ['https://www.googleapis.com/auth/drive']
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"], scopes=SCOPES)
service = build('drive', 'v3', credentials=credentials)

# Function to load the model from Google Drive
@st.cache(allow_output_mutation=True)
def load_model():
    model_file_id = '1-3_-XOrS7BUm-YsBgj5uzDzTNadx04qG'  # Replace with your model file ID
    request = service.files().get_media(fileId=model_file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    num_classes = len(class_names)
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(io.BytesIO(fh.read()), map_location=device))
    model.eval()
    return model

# Load the model
model = load_model().to(device)

# Image capture and prediction
image = st.camera_input("Capture Image")
if image:
    st.success("Image captured")
    
    # Convert BytesIO object to PIL Image
    pil_image = Image.open(image).convert("RGB")
    
    # Display the image
    st.image(pil_image)
    
    # Preprocess and predict
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_index = probabilities.argmax().item()
        predicted_class = index_to_class[predicted_index]
        predicted_probability = probabilities[predicted_index].item()
    
    # Display prediction
    st.write(f"Predicted Class: {predicted_class} (Probability: {predicted_probability:.4f})")

# Load and display the logo from Google Drive
logo_file_id = '1xIIzJsNCfuTpxAgXehy2r7QVEIsnl7Ks'  # Replace with your logo file ID
logo_request = service.files().get_media(fileId=logo_file_id)
fh_logo = io.BytesIO()
logo_downloader = MediaIoBaseDownload(fh_logo, logo_request)
done = False
while not done:
    _, done = logo_downloader.next_chunk()
fh_logo.seek(0)
logo_image = Image.open(fh_logo)
st.image(logo_image, caption='Company Logo')
