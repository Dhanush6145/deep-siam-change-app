import torch
from torch.utils.data import DataLoader
from dataset import LEVIRDataset
from models.model import SiameseChangeNet
from loss import hybrid_loss
from metrics import get_metrics
from tqdm import tqdm
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

# Dataset
train_dataset = LEVIRDataset("data/train")
val_dataset = LEVIRDataset("data/val")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# Model
model = SiameseChangeNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

epochs = 150   # YOU CONTROL HERE
print("TOTAL EPOCHS:", epochs)

best_f1 = 0

train_losses = []
val_losses = []
f1_scores = []

for epoch in range(epochs):

    # TRAIN
    model.train()
    total_loss = 0

    loop = tqdm(train_loader)

    for imgA, imgB, label in loop:
        imgA, imgB, label = imgA.to(device), imgB.to(device), label.to(device)

        optimizer.zero_grad()

        pred = model(imgA, imgB)

        # FIX SIZE MISMATCH
        pred = torch.nn.functional.interpolate(
            pred, size=label.shape[2:], mode='bilinear', align_corners=False
        )

        loss = hybrid_loss(pred, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(loss=loss.item())

    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)

    # VALIDATION
    model.eval()
    val_loss = 0
    total_f1 = 0

    with torch.no_grad():
        for imgA, imgB, label in val_loader:
            imgA, imgB, label = imgA.to(device), imgB.to(device), label.to(device)

            pred = model(imgA, imgB)

            pred = torch.nn.functional.interpolate(
                pred, size=label.shape[2:], mode='bilinear', align_corners=False
            )

            loss = hybrid_loss(pred, label)
            val_loss += loss.item()

            _, _, f1, _ = get_metrics(pred, label)
            total_f1 += f1

    val_loss /= len(val_loader)
    avg_f1 = total_f1 / len(val_loader)

    val_losses.append(val_loss)
    f1_scores.append(avg_f1)

    scheduler.step()

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")

    if avg_f1 > best_f1:
        best_f1 = avg_f1
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved!")

# PLOTS
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.savefig("loss.png")

plt.figure()
plt.plot(f1_scores, label="F1 Score")
plt.legend()
plt.savefig("f1.png")

print("Training Completed")
