import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from tqdm import tqdm
from model.msg3d import Model
import sys
from graph.ntu_rgb_d import AdjMatrixGraph
from torch.nn.utils.rnn import pad_sequence

root_dir = "C:/tanmay sop/data/data/final_keypoints"



class ActionRecognitionDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.video_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
    def __len__(self):
        return len(self.video_folders)
    
    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        kypts_path = os.path.join(video_folder, 'kypts.npy')
        label_path = os.path.join(video_folder, 'label.npy')
        
        kypts = np.load(kypts_path)
        label = np.load(label_path)
        
        # Convert numpy arrays to torch tensors
        kypts = torch.tensor(kypts, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return kypts, label


dataset = ActionRecognitionDataset(root_dir)

# Define the split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Calculate the sizes of each split
total_size = len(dataset)
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size  # To ensure the sizes sum up to total_size

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


def pad_collate(batch):
    # Unpack the batch
    (kypts, labels) = zip(*batch)

    # Pad the sequences in kypts
    kypts_padded = pad_sequence(kypts, batch_first=True)

    # Stack the labels into a tensor
    labels_tensor = torch.stack(labels, 0)

    # Add the person dimension M with size 1 if not present
    if kypts_padded.dim() == 3:  # (batch_size, T, V, C)
        kypts_padded = kypts_padded.unsqueeze(1)  # (batch_size, 1, T, V, C)

    return kypts_padded, labels_tensor

# Create data loaders for each subset
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn = pad_collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,collate_fn = pad_collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,collate_fn = pad_collate)

# Define the model
num_classes = 6  # Update this to your actual number of classes
graph_class_name = 'graph.ntu_rgb_d.AdjMatrixGraph'  # Adjust this to the actual import path of your graph class

# Import the graph class
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

# Assuming your graph class is located in 'graph.ntu_rgb_d' as 'AdjMatrixGraph'
Graph = import_class(graph_class_name)

# Create the model
model = Model(
    num_class=num_classes,
    num_point=25,
    num_person=2,
    num_gcn_scales=13,
    num_g3d_scales=6,
    graph=graph_class_name,  # Pass the string name of the graph class
    in_channels=3
)

# Load the pre-trained weights
pretrained_path = "C:/tanmay sop/msg3d-pretrained-models/pretrained-models/ntu60-xsub-bone.pt"
pretrained_dict = torch.load(pretrained_path)

# Get the model's state dict
model_dict = model.state_dict()

# Filter out the final layer parameters from the pre-trained state dict
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not k.startswith('fc')}

# Update the model's state dict with the pre-trained parameters
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# Modify the final layer to match the number of classes in your dataset
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Optionally freeze layers
for name, param in model.named_parameters():
    if not name.startswith('fc'):  # Keep the final layer unfrozen
        param.requires_grad = False

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Training Loss: {epoch_loss:.4f}')

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25)
