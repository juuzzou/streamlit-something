# app.py
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import os

# Настройки
st.title("Классификация изображений")
st.write("Загрузите изображение, и модель предскажет класс.")

# Путь к сохраненной модели
MODEL_PATH = "best_model.pth"  # <-- измените при необходимости

# Список классов — должен соответствовать train_dataset.classes
CLASSES = ["class1", "class2", "class3"]  # <-- замените на свои реальные классы

# Загружаем модель
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Трансформации (должны совпадать с теми, что использовались при обучении)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Загрузка изображения пользователем
uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    # Преобразуем изображение
    input_tensor = transform(image).unsqueeze(0)  # добавим batch dimension

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
        predicted_class = CLASSES[predicted_idx]
        confidence = probabilities[predicted_idx].item()

    st.markdown(f"### Предсказанный класс: **{predicted_class}** ({confidence*100:.2f}%)")
