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


num_classes = 60  # Update this to your actual number of classes
model = Model(
    num_class=num_classes,
    num_point=25,
    num_person=2,
    num_gcn_scales=13,
    num_g3d_scales=6,
    graph= 'graph.ntu_rgb_d.AdjMatrixGraph',
    #in_channels = 3
)



# Load the pre-trained weights
pretrained_path = "C:/tanmay sop/msg3d-pretrained-models/pretrained-models/ntu60-xsub-bone.pt"
pretrained_dict = torch.load(pretrained_path)

# Adjust the final layer to match the number of classes in your dataset
in_features = pretrained_dict['fc.weight'].shape[1]  # Get the number of input features for fc
model.fc = nn.Linear(in_features, num_classes)

# Load the pretrained weights into your model
model.load_state_dict(pretrained_dict, strict=False)

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
            print(f'Input shape: {inputs.shape}')
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()