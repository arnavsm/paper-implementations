# implementation of variational autoencoder from:
# kingma, d. p., & welling, m. (2013). auto-encoding variational bayes. arxiv:1312.6114

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # encoder: maps flattened images to hidden representation
        # input -> hidden -> latent distribution (mu, logvar)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # mean of q(z|x)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # log variance of q(z|x)
        
        # decoder: maps latent space back to image space
        # latent -> hidden -> reconstructed image
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        # pass through first hidden layer with relu activation
        h = F.relu(self.fc1(x))
        # output the parameters of the approximate posterior distribution q(z|x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        # reparameterization trick to sample z ~ q(z|x) in a differentiable way
        std = torch.exp(0.5 * logvar)       # standard deviation from log variance
        eps = torch.randn_like(std)         # random noise from standard normal
        return mu + eps * std               # shift and scale the noise
    
    def decode(self, z):
        # pass through decoder hidden layer with relu activation
        h = F.relu(self.fc2(z))
        # output reconstructed image in [0,1] range using sigmoid
        return torch.sigmoid(self.fc_out(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    # reconstruction loss: how well the reconstructed image matches the input
    # for binary images like mnist, binary cross entropy is common
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # kl divergence: measure how far q(z|x) is from the prior p(z) = n(0, i)
    # closed form for two gaussians:
    # d_kl(n(mu, sigma^2) || n(0,1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # the elbo is: expected reconstruction log likelihood - kl divergence
    # since we want to minimize negative elbo, we sum recon_loss + kl_loss
    return recon_loss + kl_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE().to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# load mnist dataset and normalize to [0,1]
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


epochs = 10
vae.train()
for epoch in range(epochs):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        # flatten the image from (batch, 1, 28, 28) to (batch, 784)
        data = data.view(-1, 784).to(device)
        
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f"epoch {epoch+1}, loss: {train_loss / len(train_loader.dataset):.4f}")


vae.eval()
with torch.no_grad():
    # sample from standard normal prior p(z)
    z = torch.randn(64, 20).to(device)
    samples = vae.decode(z).cpu()
    samples = samples.view(-1, 1, 28, 28)

    import matplotlib.pyplot as plt
    grid_img = torch.cat([samples[i] for i in range(64)], dim=2).squeeze(0)
    plt.imshow(grid_img, cmap="gray")
    plt.axis("off")
    plt.show()