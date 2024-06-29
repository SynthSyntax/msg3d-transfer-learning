import os
import numpy as np
import torch
from torch.utils.data import Dataset

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


# Example usage of the dataset
root_dir = "C:/tanmay sop/data/data/final_keypoints"
dataset = ActionRecognitionDataset(root_dir, max_frames=180)

# # Example of accessing a datapoint
# sample_kypts, sample_label = dataset[0]
# print(f"Sample keypoints shape: {sample_kypts.shape}")  # Expected output: [180, 25, 3]
# print(f"Sample label: {sample_label}")

for kypts, label in dataset:
    print(f"Sample keypoints shape: {kypts.shape}")  # Expected output: [180, 25, 3]
    print(f"Sample label: {label}")
