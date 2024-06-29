import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from model.msg3d import Model
import os
import numpy as np
from tqdm import tqdm
from graph.ntu_rgb_d import AdjMatrixGraph
from utils import count_params

root_dir = "C:/tanmay sop/data/data/final_keypoints"



class ActionRecognitionDataset(Dataset):
    def __init__(self, root_dir, max_frames=180):
        self.root_dir = root_dir
        self.max_frames = max_frames
        self.video_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
    def __len__(self):
        return len(self.video_folders)
    
    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        kypts_path = os.path.join(video_folder, 'kypts.npy')
        label_path = os.path.join(video_folder, 'label.npy')
        
        kypts = np.load(kypts_path)
        label = np.load(label_path)
        
        # Ensure all kypts have the same shape [max_frames, 25, 3]
        if kypts.shape[0] > self.max_frames:
            # Cut to max_frames frames
            kypts = kypts[:self.max_frames, :, :]
        elif kypts.shape[0] < self.max_frames:
            # Interpolate to max_frames frames
            kypts = self.interpolate_frames(kypts, self.max_frames)
        
        # Convert numpy arrays to torch tensors
        kypts = torch.tensor(kypts, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return kypts, label
    
    def interpolate_frames(self, kypts, target_frames):
        num_frames = kypts.shape[0]
        indices = np.linspace(0, num_frames - 1, target_frames, dtype=int)
        interpolated_kypts = np.zeros((target_frames, kypts.shape[1], kypts.shape[2]), dtype=np.float32)
        for i, idx in enumerate(indices):
            interpolated_kypts[i, :, :] = kypts[idx, :, :]
        return interpolated_kypts

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

# Optionally, create data loaders for each subset
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    

# sample_kypts, sample_label = dataset[0]
# print(f"Sample keypoints shape: {sample_kypts.shape}")  
# print(f"Sample label: {sample_label}")



# Load your dataset


num_classes = 2  # Update this to your actual number of classes
model = Model(
    num_class=num_classes,
    num_point=25,
    num_person=2,
    num_gcn_scales=13,
    num_g3d_scales=6,
    graph= 'graph.ntu_rgb_d.AdjMatrixGraph',
    
)



# Load the pre-trained weights
pretrained_path = "C:/tanmay sop/msg3d-pretrained-models/pretrained-models/ntu60-xsub-bone.pt"
pretrained_dict = torch.load(pretrained_path)

# Remove the final layer weights from the state dictionary
pretrained_dict.pop('fc.weight', None)
pretrained_dict.pop('fc.bias', None)


#print(model.modules)
# Load the pretrained weights into your model
model.load_state_dict(pretrained_dict, strict=False)
model.fc = nn.Linear(384, 2)


for param in model.parameters():
    param.requires_grad = False

# Ensure the new final layer's weights are trainable
for param in model.fc.parameters():
    param.requires_grad = True




# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # Perform the required transformations
            inputs = inputs.view(inputs.size(0), inputs.size(1), inputs.size(2), 1, inputs.size(3))  # [batch_size, 180, 25, 1, 3]
            inputs = inputs.repeat(1, 1, 1, 2, 1)  # Repeat to add person dimension: [batch_size, 180, 25, 2, 3]
            inputs = inputs.permute(0, 4, 1, 2, 3)  # [batch_size, 3, 180, 25, 2]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss /= len(train_loader.dataset)
        train_accuracy = correct_train / total_train

        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # Save the model if validation loss has decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

def evaluate(model, data_loader, criterion, device):
    model.eval()
    eval_loss = 0.0
    correct_eval = 0
    total_eval = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # Perform the required transformations
            inputs = inputs.view(inputs.size(0), inputs.size(1), inputs.size(2), 1, inputs.size(3))  # [batch_size, 180, 25, 1, 3]
            inputs = inputs.repeat(1, 1, 1, 2, 1)  # Repeat to add person dimension: [batch_size, 180, 25, 2, 3]
            inputs = inputs.permute(0, 4, 1, 2, 3)  # [batch_size, 3, 180, 25, 2]

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            eval_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_eval += (predicted == labels).sum().item()
            total_eval += labels.size(0)
    
    eval_loss /= len(data_loader.dataset)
    eval_accuracy = correct_eval / total_eval
    
    return eval_loss, eval_accuracy

# Define your criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Set the number of epochs and device
num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Start training
train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

# Optionally, evaluate on the test set
test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
