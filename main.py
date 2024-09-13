import torch
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import v2 as transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import time
from torchvision.models import ResNet50_Weights


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
train_transforms = transforms.Compose([
    transforms.Lambda(lambda img: add_padding(img, 256)),
    transforms.CenterCrop(224),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Cargar datos y obtener nombres de archivos
def load_data(data_dir, batch_size):
    dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    file_names = [os.path.basename(sample[0]) for sample in dataset.samples]
    return dataloader, file_names

# Configuración del entrenamiento
def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloaders:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_end = time.time()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloaders)}, Time: {epoch_end - epoch_start:.2f} sec')
    total_time = time.time() - start_time
    print(f'Total training time: {total_time:.2f} sec')

from torchvision.models import ResNet34_Weights

# Inicializar modelo
def initialize_model(num_classes):
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# Solicitar configuraciones
learning_rate = 0.001
momentum = 0.85
batch_size = 64

# Configuración de GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carga de datos
data_dir = '/content/drive/MyDrive/Proyectos/Deteccion Arte/Arte Base'
dataloaders, image_file_names = load_data(data_dir, batch_size)
num_classes = len(dataloaders.dataset.classes)
model = initialize_model(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


# Imprimir los nombres de los archivos en líneas de 5 y contar los archivos
print("Imágenes a ser procesadas:")
for i in range(0, len(image_file_names), 5):
    print(', '.join(image_file_names[i:i+5]))
print("\nTotal de imágenes:", len(image_file_names))


# Imprimir configuraciones de entrenamiento
learning_rate = optimizer.param_groups[0]['lr']
momentum = optimizer.param_groups[0]['momentum']
batch_size = dataloaders.batch_size
print(f"\nLearning Rate: {learning_rate}, Momentum: {momentum}, Batch Size: {batch_size}")



# Define the same transformation used during training
def add_padding(img, size):
    old_width, old_height = img.size
    longest_edge = max(old_width, old_height)
    horizontal_padding = (longest_edge - old_width) / 2
    vertical_padding = (longest_edge - old_height) / 2
    padded_img = Image.new("RGB", (longest_edge, longest_edge), color=(0, 0, 0))
    padded_img.paste(img, (int(horizontal_padding), int(vertical_padding)))
    return padded_img.resize((size, size), Image.LANCZOS)

test_transforms = transforms.Compose([
    transforms.Lambda(lambda img: add_padding(img, 256)),
    transforms.CenterCrop(224),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model
model_path = r'C:\Users\Karim\Downloads\Weights_ResNet50_Full layers_v3.pth'
model = models.resnet50()  # Initialize the same architecture
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 14)  # Make sure to set the correct number of output classes
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

# Load and transform the image
image = st.camera_input("Take a picture")
image_path = 'path_to_your_test_image.jpg'
image = Image.open(image_path)
image = test_transforms(image)
image = image.unsqueeze(0)  # Add batch dimension

# Predict
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    predicted_class = dataloaders.dataset.classes[predicted.item()]
    print(f"Predicted class: {predicted_class}")
