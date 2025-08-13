# implementation of variational autoencoder from:
# kingma, d. p., & welling, m. (2013). auto-encoding variational bayes. arxiv:1312.6114
## using images

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import os

class vae(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(vae, self).__init__()
        
        # encoder maps input images to parameters of approximate posterior q(z|x)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # mean of q(z|x)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # log variance of q(z|x)
        
        # decoder maps latent variables back to image space
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)     # standard deviation from log variance
        eps = torch.randn_like(std)       # noise from standard normal
        return mu + eps * std             # sample using reparameterization trick
    
    def decode(self, z):
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc_out(h))  # output pixel values in [0, 1]
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    # reconstruction loss measures how close output is to input
    # for mnist, binary cross entropy is appropriate since pixels are in [0,1]
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # kl divergence term keeps q(z|x) close to prior p(z) = n(0, i)
    # this is the closed form for two diagonal gaussians
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # total loss is negative elbo = reconstruction loss + kl divergence
    return recon_loss + kl_loss

# set device and hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 20
vae_model = vae(latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)

# load mnist dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# create output folder for samples
os.makedirs("vae_samples", exist_ok=True)

# train the model
epochs = 10
vae_model.train()
for epoch in range(epochs):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, 784).to(device)
        
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae_model(data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    avg_loss = train_loss / len(train_loader.dataset)
    print(f"epoch {epoch+1}, average loss: {avg_loss:.4f}")
    
    # sample images from prior and save for this epoch
    vae_model.eval()
    with torch.no_grad():
        z = torch.randn(64, latent_dim).to(device)       # sample from p(z)
        samples = vae_model.decode(z).cpu()
        samples = samples.view(-1, 1, 28, 28)
        utils.save_image(samples, f"vae_samples/sample_epoch_{epoch+1}.png", nrow=8)
    vae_model.train()

# display the last saved sample grid
img = plt.imread(f"vae_samples/sample_epoch_{epochs}.png")
plt.imshow(img)
plt.axis("off")
plt.show()