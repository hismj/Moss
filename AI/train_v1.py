import os
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
from model import shufflenet_v2_x2_0


class APKImageDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = []
        self.label_map = {'benign': 0, 'scam': 1}

        with open(label_file, 'r') as f:
            for line in f:
                img_name, label = line.strip().split()
                self.img_labels.append((img_name, self.label_map[label]))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, label = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return {
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix
    }

# Scam and benign datasets prepared
scam_img_dir = 'img_scam'
scam_label_file = 'scam.txt'
benign_img_dir = 'img_benign'
benign_label_file = 'benign.txt'

# Create dataset for both classes
scam_dataset = APKImageDataset(scam_img_dir, scam_label_file, transform)
benign_dataset = APKImageDataset(benign_img_dir, benign_label_file, transform)

# Combine datasets
full_dataset = torch.utils.data.ConcatDataset([scam_dataset, benign_dataset])

# Split into training, validation, and test sets (8:1:1 ratio)
train_size = int(0.8 * len(full_dataset))
val_test_size = len(full_dataset) - train_size
val_size = int(0.5 * val_test_size)
test_size = val_test_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Create data loaders
batch_size = 24
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Train and evaluate the model
model = shufflenet_v2_x2_0()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_losses = []

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

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Save the model
model_path = 'shufflenet_v2_x2_0_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Plotting the training loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model on the validation set
val_results = evaluate_model(model, val_loader, criterion)
print("Validation Results:", val_results)

# Evaluate the model on the test set
test_results = evaluate_model(model, test_loader, criterion)
print("Test Results:", test_results)
