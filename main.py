import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io
import pandas as pd

# Define class names and the path for the model
class_names = {'CLUSTER': 0, 'DANGLER': 1, 'KIT COPETE': 2, 'KIT DANG BOTADERO': 3,
               'MANTELETA': 4, 'MENU': 5, 'MP': 6, 'PC': 7, 'POSTER': 8,
               'PRECIADOR': 9, 'REFRICALCO': 10, 'STICKER': 11, 'STOPPER': 12, 'V UN': 13}
index_to_class = {v: k for k, v in class_names.items()}
model_path = 'models/Full_ResNet50_Ful layers_v3.pth'

# Load the model with new Streamlit caching
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.eval()
    model.to(device)
    return model, device

model, device = load_model()

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

def predict(image_file):
    image = Image.open(image_file).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        # Get top three predictions
        top_probs, top_lbls = torch.topk(probabilities, 3)
        predicted_classes = [(index_to_class[idx.item()], prob.item()) for idx, prob in zip(top_lbls, top_probs)]
    return predicted_classes

# Streamlit UI
st.title('Image Classification App')


#uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "JPG"], key="file_uploader")
uploaded_file = st.camera_input("Take a picture", key="camera_input")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    if st.button('Predict'):
        uploaded_file.seek(0)  # Reset file pointer
        predictions = predict(uploaded_file)
        # Display predictions as a table
        df = pd.DataFrame(predictions, columns=['Class', 'Probability'])
        st.table(df)
