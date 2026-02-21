import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
from sklearn.metrics import f1_score, classification_report
import sys

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Use a tiny subset to show quick results if requested
print("Loading subsets of CIFAR-10 for quick evaluation demonstration...")
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# subset for fast execution
train_subset = torch.utils.data.Subset(train_dataset, range(5000)) # 10%
test_subset = torch.utils.data.Subset(test_dataset, range(1000))  # 10%

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- ARCHITECTURES ---
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*32*3, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(self.flatten(x))))))

class AlexNetCIFAR(nn.Module):
    def __init__(self):
        super(AlexNetCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

class TinyVGG(nn.Module):
    def __init__(self):
        super(TinyVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_eval(model, name, epochs=2):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n--- {name} (Params: {count_parameters(model):,}) ---")
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} Loss: {running_loss/len(train_loader):.4f}")
    
    train_time = time.time() - start_time
    
    # Evaluate
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Eval -> Accuracy: {acc:.2f}%, F1-Score: {f1:.4f}, Time taken: {train_time:.2f}s")
    return acc, f1, train_time

if __name__ == "__main__":
    print(f"{'Model':<15} {'Parameters':>12} {'Train Time':>12} {'Accuracy':>12} {'F1-Score':>12}")
    
    acc1, f11, t1 = train_and_eval(SimpleNN(), "SimpleNN", epochs=2)
    acc2, f12, t2 = train_and_eval(TinyVGG(), "TinyVGG", epochs=2)
    acc3, f13, t3 = train_and_eval(AlexNetCIFAR(), "AlexNet", epochs=2)
    
    print("\n--- Summary (Subsampled 10% Dataset for Speed) ---")
    print(f"{'Model':<15} {'Parameters':>12} {'Train Time':>12} {'Accuracy':>12} {'F1-Score':>12}")
    print(f"{'SimpleNN':<15} {1707274:>12,} {t1:>11.2f}s {acc1:>11.2f}% {f11:>12.4f}")
    print(f"{'TinyVGG':<15} {591274:>12,} {t2:>11.2f}s {acc2:>11.2f}% {f12:>12.4f}")
    print(f"{'AlexNet':<15} {35855178:>12,} {t3:>11.2f}s {acc3:>11.2f}% {f13:>12.4f}")
