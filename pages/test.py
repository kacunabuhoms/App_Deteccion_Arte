import torch
from torchvision import models, transforms
from PIL import Image

# Establecer el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Cargar el modelo
def load_model(model_path):
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    num_classes = len(class_names)  # Ajustar al número de clases
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

model_path = 'models/Full_ResNet50_Ful layers_v3.pth'
model = load_model(model_path)

# Función para predecir y devolver la clase y la probabilidad
def predict_class(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class_index = probabilities.argmax().item()
        predicted_class_name = index_to_class[predicted_class_index]
        predicted_probability = probabilities[predicted_class_index].item()

    return predicted_class_name, predicted_probability

# Ejemplo de uso
image_path = 'test/cluster ejem.png'
predicted_class_name, predicted_probability = predict_class(image_path)
print(f'Clase Predicha: {predicted_class_name}, Probabilidad: {predicted_probability:.4f}')