 
# Import

 
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import time
from sklearn.metrics import precision_score, recall_score, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())  

 
# Customize image

 
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

 
# Transform image

 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

 
# Load images

 
# Load dataset
cur_dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(cur_dir, 'CatsDogsDataset', 'train')
root  = os.path.join(cur_dir, 'CatsDogsDataset', 'test')
dataset = CustomImageDataset(img_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
test_dataset = datasets.ImageFolder(root, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Get class names (folder names) for reference
class_names = test_dataset.classes
print(f'Class names: {class_names}')


 
# Generator

 
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(100, 512 * 7 * 7) 
        self.resblock = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), # Output should be 224x224x3
            nn.Tanh()  # Normalize to [-1, 1] for image generation
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 512, 7, 7)  # Reshape to start with 7x7x512 feature map
        return self.resblock(x)

 
# Discriminator

 
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 224 -> 112
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 112 -> 56
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 56 -> 28
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 28 -> 14
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # 14 -> 7
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 1),  # Adjusted for the size after convolutions (7x7x1024)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)

 
# ClusterHead

 
class ClusterHead(nn.Module):
    def __init__(self, num_clusters=2):
        super(ClusterHead, self).__init__()
        self.fc = nn.Linear(224*224*3, num_clusters)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

 
# Evaluation model

 
# Function to evaluate the model on test data
def evaluate_model(generator, cluster_head, test_loader, device):
    generator.eval()
    cluster_head.eval()

    all_pred_labels = []
    all_true_labels = []

    start_time = time.time()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            true_labels = labels.numpy()  # True labels (folder names) as integers

            # Generate fake images
            noise = torch.randn(images.size(0), 100).to(device)
            fake_images = generator(noise)

            # Get cluster predictions
            cluster_logits = cluster_head(fake_images)
            _, predicted_labels = torch.max(cluster_logits, 1)  # Predicted cluster labels

            # Append to lists
            all_pred_labels.extend(predicted_labels.cpu().numpy())
            all_true_labels.extend(true_labels)

    # Calculate metrics
    precision = precision_score(all_true_labels, all_pred_labels, average='weighted')
    recall = recall_score(all_true_labels, all_pred_labels, average='weighted')
    f1 = f1_score(all_true_labels, all_pred_labels, average='weighted')
    accuracy = (np.array(all_pred_labels) == np.array(all_true_labels)).mean()

    end_time = time.time()
    runtime = end_time - start_time

    return precision, recall, f1, accuracy, runtime

 
def plot_evaluation_metrics(precision, recall, f1, accuracy):
    # Metrics names
    metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    values = [precision, recall, f1, accuracy]

    # Create the bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylim(0, 1)  # Set the y-axis limits to be between 0 and 1
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Clustering Evaluation Metrics')

    # Display the score on top of each bar
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

 
# Train model

 
# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)
cluster_head = ClusterHead(num_clusters=2).to(device)

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss functions
criterion = nn.BCELoss()

# Training loop
num_epochs = 36

# Track metrics for plotting
d_losses = []
g_losses = []
cluster_losses = []

for epoch in range(num_epochs):
    d_loss_epoch = 0.0
    g_loss_epoch = 0.0
    cluster_loss_epoch = 0.0
    num_batches = 0

    for i, images in enumerate(dataloader):
        # Move images to the GPU
        images = images.to(device)

        # Train discriminator
        real_labels = torch.ones(images.size(0), 1).to(device)
        fake_labels = torch.zeros(images.size(0), 1).to(device)

        # Train with real images
        optimizer_d.zero_grad()
        real_output = discriminator(images)
        d_loss_real = criterion(real_output, real_labels)

        # Train with fake images generated by the generator
        noise = torch.randn(images.size(0), 100).to(device)
        fake_images = generator(noise)
        print(fake_images.shape)
        fake_output = discriminator(fake_images.detach())
        d_loss_fake = criterion(fake_output, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        # Train generator
        optimizer_g.zero_grad()
        fake_output = discriminator(fake_images)
        g_loss = criterion(fake_output, real_labels)

        # Cluster loss
        cluster_logits = cluster_head(fake_images)
        cluster_loss = torch.mean((cluster_logits - real_labels) ** 2)

        total_loss = g_loss + cluster_loss
        total_loss.backward()
        optimizer_g.step()

        # Track losses for each batch
        d_loss_epoch += d_loss.item()
        g_loss_epoch += g_loss.item()
        cluster_loss_epoch += cluster_loss.item()
        num_batches += 1

    # Average loss per epoch
    d_losses.append(d_loss_epoch / num_batches)
    g_losses.append(g_loss_epoch / num_batches)
    cluster_losses.append(cluster_loss_epoch / num_batches)

    print(f'Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

 
# Plotting losses
plt.figure(figsize=(10, 5))

# Plot Discriminator and Generator Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), d_losses, label='Discriminator Loss')
plt.plot(range(1, num_epochs + 1), g_losses, label='Generator Loss')
plt.title('Generator and Discriminator Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Cluster Loss
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), cluster_losses, label='Cluster Loss', color='purple')
plt.title('Cluster Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

 
# Test

 
# Evaluate the model
precision, recall, f1, accuracy, runtime = evaluate_model(generator, cluster_head, test_loader, device)

# Output the results
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'Runtime Efficiency: {runtime:.4f} seconds')

plot_evaluation_metrics(precision, recall, f1, accuracy)


