import os
import cv2
import torch
from torch.utils.data import Dataset

class LEVIRDataset(Dataset):
    def __init__(self, root_dir):
        self.A = os.path.join(root_dir, "A")
        self.B = os.path.join(root_dir, "B")
        self.L = os.path.join(root_dir, "label")

        self.files = sorted(os.listdir(self.A))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        imgA_path = os.path.join(self.A, name)
        imgB_path = os.path.join(self.B, name)
        label_path = os.path.join(self.L, name)

        imgA = cv2.imread(imgA_path)
        imgB = cv2.imread(imgB_path)
        label = cv2.imread(label_path, 0)

        # Check if image loaded
        if imgA is None or imgB is None or label is None:
            raise ValueError(f"Error loading file: {name}")

        imgA = cv2.resize(imgA, (256, 256))
        imgB = cv2.resize(imgB, (256, 256))
        label = cv2.resize(label, (256, 256))

        imgA = torch.tensor(imgA).permute(2, 0, 1).float() / 255
        imgB = torch.tensor(imgB).permute(2, 0, 1).float() / 255
        label = torch.tensor(label).unsqueeze(0).float() / 255

        return imgA, imgB, label
