import os
import argparse
import torch.optim as optim
from PIL import Image
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix
from torch.cuda import device
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Define a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APKImageDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = []

        with open(label_file, 'r') as f:
            for line in f:
                img_name, label = line.strip().split()
                self.img_labels.append((img_name, label))

        self.label_map = {'benign': 0, 'scam': 1}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, label = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # 打印裁剪前的原始图像尺寸
        print("Original image size:", image.size)

        if self.transform:
            image = self.transform(image)

        # 打印转换后的图像尺寸
        print("Transformed image size:", image.size())

        label = self.label_map[label]

        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Define the enhanced ShuffleNetV2 model with ECA module
class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        mid_channels = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.dwconv = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1,
                                groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        if self.stride == 2:
            x1 = self.conv1(x)
            x1 = self.bn1(x1)
            x1 = self.relu(x1)
            x1 = self.dwconv(x1)
            x1 = self.bn2(x1)
            x1 = self.relu(x1)
            x1 = self.conv2(x1)
            x1 = self.bn3(x1)
            x2 = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
            return torch.cat((x1, x2), dim=1)
        else:
            x1, x2 = x.chunk(2, dim=1)
            x1 = self.conv1(x1)
            x1 = self.bn1(x1)
            x1 = self.relu(x1)
            x1 = self.dwconv(x1)
            x1 = self.bn2(x1)
            x1 = self.relu(x1)
            x1 = self.conv2(x1)
            x1 = self.bn3(x1)
            return torch.cat((x1, x2), dim=1)


class ShuffleNetV2ECA(nn.Module):
    def __init__(self, stages_out_channels, num_classes=2):
        super(ShuffleNetV2ECA, self).__init__()
        self.conv1 = nn.Conv2d(3, stages_out_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stages_out_channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1], stride=2)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2], stride=2)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3], stride=2)
        self.fc = nn.Linear(stages_out_channels[3], num_classes)

    def _make_stage(self, in_channels, out_channels, stride):
        blocks = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(3):
            blocks.append(BasicBlock(out_channels, out_channels, stride=1))
        blocks.append(ECA(out_channels))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def shufflenet_v2_eca():
    stages_out_channels = [24, 48, 96, 192]
    model = ShuffleNetV2ECA(stages_out_channels, num_classes=2)  # Modify num_classes as needed
    return model


def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return {
        'loss': total_loss / len(dataloader),
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix
    }
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
torch.cuda.set_device(0)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

        # Evaluate on validation set
        eval_results = evaluate_model(model, val_loader, criterion)
        logger.info(f"Validation Results - Epoch [{epoch + 1}/{num_epochs}]: {eval_results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APK Image Classification Training")
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for optimizer")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for training and validation")
    args = parser.parse_args()

    # scam and benign datasets prepared
    scam_img_dir = 'img_scam'
    scam_label_file = 'scam.txt'
    benign_img_dir = 'img_benign'
    benign_label_file = 'benign.txt'

    # Create dataset for both classes
    scam_dataset = APKImageDataset(scam_img_dir, scam_label_file, transform)
    benign_dataset = APKImageDataset(benign_img_dir,benign_label_file,transform)
    print('数据集已经加载完成')
    # Combine datasets
    full_dataset = scam_dataset + benign_dataset

    # Split into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model, criterion, optimizer
    model = shufflenet_v2_eca()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train and evaluate the model
    train_model(model, train_loader, val_loader, criterion, optimizer, args.epochs)

