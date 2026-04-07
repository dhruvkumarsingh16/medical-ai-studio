import torch
from torchvision import datasets, transforms
from model import VAE
import torch.nn.functional as F

# Image transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load dataset
dataset = datasets.ImageFolder("dataset_folder", transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

# Model
model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 64*64), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Training loop
for epoch in range(5):
    for data, _ in loader:
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss = loss_function(recon, data, mu, logvar)
        loss.backward()
        optimizer.step()

    print("Epoch:", epoch, "Loss:", loss.item())

# Save model
torch.save(model.state_dict(), "vae.pth")
print("Model saved!")
