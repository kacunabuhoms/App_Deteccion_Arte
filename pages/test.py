import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os

# Define the path for the model
model_path = 'models/Full_ResNet50_Ful layers_v3.pth'  # Adjust path if necessary based on your repository structure

# Load the model
@st.cache(allow_output_mutation=True)
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

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class_index = probabilities.argmax().item()
        predicted_class_name = {v: k for k, v in model.class_names.items()}[predicted_class_index]
        predicted_probability = probabilities[predicted_class_index].item()
    return f'Predicted Class: {predicted_class_name} (Probability: {predicted_probability:.4f})'

# Streamlit UI
st.title('Image Classification App')

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image_path = uploaded_file.name
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(image_path, caption='Uploaded Image', use_column_width=True)
    if st.button('Predict'):
        result = predict(image_path)
        st.write(result)
