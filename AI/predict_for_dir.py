import os
import torch
from torchvision import transforms
from PIL import Image
from model import shufflenet_v2_x2_0
from pred_dex_extract import process_single_apk


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

    return predicted.item(), image

if __name__ == '__main__':
    label_map = {0: 'benign', 1: 'scam'}
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model_path = 'shufflenet_v2_x2_0_model_1.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    apk_folder = input("请输入需要预测的apk文件夹路径! :) ：")
    total_count = 0
    scam_count = 0

    for apk_file in os.listdir(apk_folder):
        if apk_file.endswith('.apk'):
            apk_path = os.path.join(apk_folder, apk_file)
            image_path = process_single_apk(apk_path)

            if image_path:
                predicted_label = predict(image_path, model, transform, device)
                total_count += 1
                if predicted_label == 1:  # 'scam' label
                    scam_count += 1
                print(f"{apk_file}: Predicted label: {label_map[predicted_label]}")
            else:
                print(f"处理APK文件 {apk_file} 时出错，请检查并重试。")

    if total_count > 0:
        scam_ratio = scam_count / total_count * 100
    else:
        scam_ratio = 0

    print(f"\n总共处理了 {total_count} 个 APK 文件。")
    print(f"Scam 文件数量：{scam_count}")
    print(f"Scam 占比：{scam_ratio:.2f}%")
