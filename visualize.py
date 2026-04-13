import torch
import matplotlib.pyplot as plt
from dataset import LEVIRDataset
from models.model import SiameseChangeNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SiameseChangeNet().to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()

data = LEVIRDataset("data/test")

imgA, imgB, label = data[0]

with torch.no_grad():
    pred = model(imgA.unsqueeze(0).to(device),
                 imgB.unsqueeze(0).to(device))
    pred = pred.squeeze().cpu().numpy()

plt.figure(figsize=(10,5))

plt.subplot(1,4,1)
plt.title("Before")
plt.imshow(imgA.permute(1,2,0))

plt.subplot(1,4,2)
plt.title("After")
plt.imshow(imgB.permute(1,2,0))

plt.subplot(1,4,3)
plt.title("Ground Truth")
plt.imshow(label.squeeze(), cmap='gray')

plt.subplot(1,4,4)
plt.title("Prediction")
plt.imshow(pred > 0.5, cmap='gray')

plt.show()
