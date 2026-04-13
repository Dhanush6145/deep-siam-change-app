import torch
import torch.nn as nn
import torch.optim as optim
from dataset import ChangeDataset
from models.model import SiameseChangeNet
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# TRANSFORM (FIXED)
# -------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# -------------------------
# DATA
# -------------------------
train_dataset = ChangeDataset("data/train", transform=transform)
val_dataset = ChangeDataset("data/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# -------------------------
# MODEL
# -------------------------
model = SiameseChangeNet().to(DEVICE)

# -------------------------
# LOSS (UPGRADED)
# -------------------------
bce = nn.BCEWithLogitsLoss()

def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1.0
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))

def loss_fn(pred, target):
    return bce(pred, target) + dice_loss(pred, target)

# -------------------------
# OPTIMIZER
# -------------------------
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -------------------------
# TRAIN LOOP
# -------------------------
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for img1, img2, label in train_loader:
        img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)

        output = model(img1, img2)
        loss = loss_fn(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# SAVE MODEL
torch.save(model.state_dict(), "model.pth")
print("Model saved!")
