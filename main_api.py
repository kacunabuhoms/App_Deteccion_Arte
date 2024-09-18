from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import torch
from torchvision import models, transforms
import os

app = FastAPI()

# Configuración del modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Diccionario de clases
class_names = {
    'CLUSTER': 0, 'DANGLER': 1, 'KIT COPETE': 2, 'KIT DANG BOTADERO': 3,
    'MANTELETA': 4, 'MENU': 5, 'MP': 6, 'PC': 7, 'POSTER': 8,
    'PRECIADOR': 9, 'REFRICALCO': 10, 'STICKER': 11, 'STOPPER': 12, 'V UN': 13
}

# Carga el modelo de PyTorch
def load_model():
    model_path = os.path.join('models', 'Full_ResNet50_Ful layers_v3.pth')
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model

def load_model_safe(model_path, device):
    # Configura un entorno seguro para la carga
    with open(model_path, 'rb') as f:
        model = torch.load(f, map_location=device)

    model.to(device)
    model.eval()
    return model

model = load_model_safe(os.path.join('models', 'Full_ResNet50_Ful layers_v3.pth'), device)


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

# Endpoint para hacer predicciones
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(BytesIO(image_data)).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_index = probabilities.argmax().item()
        predicted_class = class_names[predicted_index]
        predicted_probability = probabilities[predicted_index].item()

    return {"class_name": predicted_class, "probability": predicted_probability}

# Ejecutar el servidor usando Uvicorn si este archivo se ejecuta directamente
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
