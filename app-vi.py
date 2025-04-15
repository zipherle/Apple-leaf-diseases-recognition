import torch
import torch.nn as nn
from torchvision import transforms, models
import streamlit as st
from PIL import Image
class_name = ['Lá bị thối', 'Lá khỏe mạnh', 'Lá bị đốm', 'Lá bị ghẻ']
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
if 'model' not in st.session_state:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, len(class_name))
    model.load_state_dict(torch.load('model/cnn-model2.pth'))
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    st.session_state.model = model
st.title("Phát hiện bệnh cho cây")
uploaded_image = st.file_uploader("Tải hình ảnh lên", type=["jpg", "png", "jpeg"])
if uploaded_image is not None:
    img = Image.open(uploaded_image)
    last_img = img
    img = img.resize((64, 64))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    st.image(last_img)
    with torch.no_grad():
        output = st.session_state.model(img)
        _, predicted = torch.max(output, 1)
        predicted_class = class_name[predicted.item()]
        st.write(f'Dự đoán: {predicted_class}')
