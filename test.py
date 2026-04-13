import torch
from torch.utils.data import DataLoader
from dataset import LEVIRDataset
from models.model import SiameseChangeNet
from metrics import compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")

model = SiameseChangeNet().to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()

print("Loading dataset...")

test_data = LEVIRDataset("data/test")
loader = DataLoader(test_data, batch_size=1)

print("Total test samples:", len(loader))  # 🔥 IMPORTANT

total_iou, total_f1 = 0, 0

with torch.no_grad():
    for i, (imgA, imgB, label) in enumerate(loader):

        print(f"Processing sample {i}")  # 🔥 DEBUG

        imgA, imgB, label = imgA.to(device), imgB.to(device), label.to(device)

        pred = model(imgA, imgB)
        pred = (pred > 0.5).float()

        iou, f1 = compute_metrics(pred, label)

        print("IoU:", iou, "F1:", f1)  # 🔥 DEBUG

        total_iou += iou
        total_f1 += f1

if len(loader) > 0:
    print("Average IoU:", total_iou / len(loader))
    print("Average F1:", total_f1 / len(loader))
else:
    print("Dataset is EMPTY ❌")
