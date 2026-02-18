import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# Import SpikingJelly packages
from spikingjelly.activation_based import neuron, functional

# ----------------------------
# 1. Data Loading and Preprocessing
# ----------------------------
cur_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(cur_dir, 'CatsDogsDataset', 'train')
test_dir  = os.path.join(cur_dir, 'CatsDogsDataset', 'test')

train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset  = datasets.ImageFolder(test_dir, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 2. Define the Convolutional SNN Model
# ----------------------------
class ConvSNN(nn.Module):
    def __init__(self, time_window=20):
        super(ConvSNN, self).__init__()
        self.time_window = time_window

        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.lif1  = neuron.LIFNode(tau=2.0)
        self.pool1 = nn.MaxPool2d(2)  # 64x64 -> 32x32

        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.lif2  = neuron.LIFNode(tau=2.0)
        self.pool2 = nn.MaxPool2d(2)  # 32x32 -> 16x16

        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.lif3  = neuron.LIFNode(tau=2.0)
        self.pool3 = nn.MaxPool2d(2)  # 16x16 -> 8x8

        # Fully connected layer for binary classification.
        self.fc = nn.Linear(128 * 8 * 8, 1)
        
    def forward(self, x):
        batch_size = x.size(0)
        out_sum = 0

        for t in range(self.time_window):
            # First block.
            out = self.conv1(x)
            out = self.bn1(out)
            out = F.relu(out)
            if not hasattr(self.lif1, 'v') or self.lif1.v is None or not isinstance(self.lif1.v, torch.Tensor) or self.lif1.v.shape != out.shape:
                self.lif1.v = torch.zeros_like(out)
            else:
                self.lif1.v = self.lif1.v.detach()
            out = self.lif1(out)
            out = self.pool1(out)
            
            # Second block.
            out = self.conv2(out)
            out = self.bn2(out)
            out = F.relu(out)
            if not hasattr(self.lif2, 'v') or self.lif2.v is None or not isinstance(self.lif2.v, torch.Tensor) or self.lif2.v.shape != out.shape:
                self.lif2.v = torch.zeros_like(out)
            else:
                self.lif2.v = self.lif2.v.detach()
            out = self.lif2(out)
            out = self.pool2(out)
            
            # Third block.
            out = self.conv3(out)
            out = self.bn3(out)
            out = F.relu(out)
            if not hasattr(self.lif3, 'v') or self.lif3.v is None or not isinstance(self.lif3.v, torch.Tensor) or self.lif3.v.shape != out.shape:
                self.lif3.v = torch.zeros_like(out)
            else:
                self.lif3.v = self.lif3.v.detach()
            out = self.lif3(out)
            out = self.pool3(out)
            
            out = out.view(batch_size, -1)
            out = self.fc(out)
            
            if t < self.time_window - 1:
                out_sum = out_sum + out.detach()
            else:
                out_sum = out_sum + out

        out_avg = out_sum / self.time_window
        return out_avg

model_snn = ConvSNN(time_window=20).to(device)

# ----------------------------
# 3. Train the Convolutional SNN
# ----------------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model_snn.parameters(), lr=5e-5)


num_epochs = 25
train_loss_history = []
valid_acc_history = []

for epoch in range(num_epochs):
    model_snn.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model_snn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        # Optionally reset network state:
        # functional.reset_net(model_snn)
    
    epoch_loss = running_loss / len(train_dataset)
    train_loss_history.append(epoch_loss)
    
    # Evaluate validation accuracy at the end of the epoch
    model_snn.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model_snn(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()
            correct += (preds.cpu() == labels.cpu().int()).sum().item()
            total += labels.size(0)
    epoch_acc = correct / total
    valid_acc_history.append(epoch_acc)
    
    
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")
        
   


# ----------------------------
# 4. Evaluate the Convolutional SNN on Test Set
# ----------------------------

start_time = time.time()
model_snn.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model_snn(inputs)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int().cpu().numpy().flatten()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())


acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, zero_division=0)
rec  = recall_score(all_labels, all_preds, zero_division=0)
f1   = f1_score(all_labels, all_preds, zero_division=0)
end_time = time.time()
running_time = end_time - start_time

print("\nConvolutional SNN Performance on Test Set:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"Execution time: {running_time:.2f} seconds")

# ----------------------------
# 5. Visualization of Training Loss and Validation Accuracy
# ----------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_loss_history, marker='o', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), valid_acc_history, marker='o', color='orange', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
