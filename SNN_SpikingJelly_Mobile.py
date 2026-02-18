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

# Import SpikingJelly packages
from spikingjelly.activation_based import ann2snn, functional

# ----------------------------
# 1. Data Loading and Preprocessing
# ----------------------------
# Setup paths (adjust as needed)
cur_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(cur_dir, 'CatsDogsDataset', 'train')
test_dir  = os.path.join(cur_dir, 'CatsDogsDataset', 'test')

# Define data transforms (similar to your ImageDataGenerator setup)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Create datasets and dataloaders.
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset  = datasets.ImageFolder(test_dir, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)



# ----------------------------
# 2. Define and Train the ANN (Conventional CNN)
# ----------------------------

ann_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("This running time is using: ", device)
ann_model.to(device)


#device = torch.device("cpu")

ann_model.classifier[1] = torch.nn.Linear(
        in_features=ann_model.classifier[1].in_features,  # as in the original output layer
        out_features=2).to(device)

for param in ann_model.parameters():
    param.requires_grad = False

# Only the classifier layers will be trained
for param in ann_model.classifier.parameters():
    param.requires_grad = True

criterion = nn.BCELoss()
optimizer = optim.Adam(ann_model.parameters(), lr=5e-4)

m = nn.Softmax(dim = 1) #softmax for squashing the processing of the output into [0, 1]


num_epochs = 10

ANN_CPU_PATH = "./mobile_base_cpu.pt"
ANN_CUDA_PATH = "./mobile_base_cuda.pt"


if (os.path.exists(ANN_CPU_PATH)):
	device = torch.device("cpu")
	ann_model = torch.load(ANN_CPU_PATH, weights_only=False)

elif (os.path.exists(ANN_CUDA_PATH) and torch.cuda.is_available()):
    ann_model = torch.load(ANN_CUDA_PATH, weights_only=False)

else:
    
    for epoch in range(num_epochs):
        ann_model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            # Convert labels to float tensor and reshape to (batch, 2)
            labels = nn.functional.one_hot(labels, num_classes = 2).to(device).float()
            #labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = m(ann_model(inputs)).to(device).float()
            #print(f"labels' shape is {labels.shape}, outputs' shape is {outputs.shape}.")
            loss = criterion(outputs, labels) # The average of the loss 
            #loss.requires_grad = True
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f}")
    torch.save(ann_model, ANN_CUDA_PATH if torch.cuda.is_available() else ANN_CPU_PATH)
    

# Evaluate ANN on test set
ann_model.eval()
ann_preds = []
ann_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = torch.max(m(ann_model(inputs)), 1).indices.cpu().numpy()  #shape: (batch_size, )
        ann_preds.extend(outputs)
        ann_labels.extend(labels.numpy())
acc_ann = accuracy_score(ann_labels, ann_preds)
prec_ann = precision_score(ann_labels, ann_preds, zero_division=0)
recall_ann = recall_score(ann_labels, ann_preds, zero_division=0)
f1_ann = f1_score(ann_labels, ann_preds, zero_division=0)

print("\nANN Performance:")
print(f"Accuracy: {acc_ann:.4f}, Precision: {prec_ann:.4f}, Recall: {recall_ann:.4f}, F1-score: {f1_ann:.4f}")


    
# ----------------------------
# 3. Convert ANN to SNN using SpikingJelly
# ----------------------------
# The conversion process replaces the ReLU activations with spiking neurons.
# Here we use SpikingJelly's ann2snn.convert. The time window (T) defines how many simulation timesteps we run.
T = 50 # simulation time steps

# Convert the ANN model to an SNN model.
# Note: ann2snn.convert will replace activations (like F.relu) with spiking neuron modules.
model_converter = ann2snn.Converter(mode='max', dataloader=train_loader)
snn_model = model_converter(ann_model)
snn_model.to(device)
snn_model.eval()



# ----------------------------
# 4. Evaluate the Converted SNN (Updated)
# ----------------------------
start_time = time.time()
snn_preds = []
functional.reset_net(snn_model)
snn_labels = []  
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        batch_outputs = []
        # Reset the network's state for each batch.
        functional.reset_net(snn_model)
        # Run the SNN over T timesteps.
        for t in range(T):
            out_t = snn_model(inputs) # shape: (batch_size, 2) 
            # print("out_t shape is:", out_t.shape)
            out_t = torch.max(m(out_t), 1).indices.cpu().flatten() # shape(batch_size, )
            batch_outputs.append(out_t)
            
        outputs_time = torch.stack(batch_outputs, dim=0)  # shape: (T, batch_size)
        preds = outputs_time.float().mean(dim=0)  # average over timesteps -> (batch_size, )
        snn_preds.extend(preds)
        snn_labels.extend(labels.numpy())

acc_snn = accuracy_score(snn_labels, snn_preds)
prec_snn = precision_score(snn_labels, snn_preds, zero_division=0)
recall_snn = recall_score(snn_labels, snn_preds, zero_division=0)
f1_snn = f1_score(snn_labels, snn_preds, zero_division=0)

end_time = time.time()
running_time = end_time - start_time

print(f"Total execution time: {running_time:.2f} seconds")
print("\nSNN Performance (Converted from ANN):")
print(f"Accuracy: {acc_snn:.4f}, Precision: {prec_snn:.4f}, Recall: {recall_snn:.4f}, F1-score: {f1_snn:.4f}")


# ----------------------------
# 5. (Optional) Plot a Sample's Output Over Time
# ----------------------------
# For one batch from the validation set, plot the output of the SNN over the simulation timesteps.

""" with torch.no_grad():
    # Get one batch from the test set.
    sample_inputs, _ = next(iter(test_loader))
    sample_inputs = sample_inputs.to(device)  # shape: (batch, channels, H, W)
    
    sample_outputs_list = []
    # Reset the SNN state before simulation.
    functional.reset_net(snn_model)
    
    # Iterate over T timesteps.
    for t in range(T):
        # Each call receives a 4D tensor.
        out_t = snn_model(sample_inputs)  # shape: (batch, 1)
        sample_outputs_list.append(out_t)
    
    # Stack outputs along a new time dimension: shape (T, batch, 1)
    sample_outputs = torch.stack(sample_outputs_list, dim=0)
    
    # For plotting, take the output of the first sample over time.
    sample_output_values = sample_outputs[:, 0, 0].cpu().numpy()

plt.figure(figsize=(10, 5))
plt.plot(range(T), sample_output_values, marker='o')
plt.xlabel("Timestep")
plt.ylabel("Output Value")
plt.title("SNN Output Over Time (Batch 0 as a Sample)")
plt.grid(True)
plt.show() """