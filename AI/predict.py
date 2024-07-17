import os
import tempfile
import torch
from torchvision import transforms
from PIL import Image
from AI.model import shufflenet_v2_x2_0
from AI.pred_dex_extract import process_single_apk

def load_model(model_path, device):
    model = shufflenet_v2_x2_0()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def predict(image_path, model, transform, device):
    image = preprocess_image(image_path, transform)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()

def apk_predict(apk_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model_path = 'AI/shufflenet_v2_x2_0_model_1.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    # Save uploaded file to a temporary location

    image_path = process_single_apk(apk_path)
    if image_path:
        predicted_label = predict(image_path, model, transform, device)
        return predicted_label
    else:
        return "处理APK文件时出错，请重试。"
