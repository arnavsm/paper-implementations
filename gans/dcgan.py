import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.0002
batch_size = 128
image_size = 64  
channels_img = 1 
latent_dim = 100 
n_epochs = 25


class Discriminator(nn.Module):
    def __init__(self, channels_img):
        super(Discriminator, self).__init__()
        # input: N x channels_img x 64 x 64
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, 64, kernel_size=4, stride=2, padding=1), # -> 64 x 32 x 32
            nn.LeakyReLU(0.2),
            self._block(64, 128, 4, 2, 1),    # -> 128 x 16 x 16
            self._block(128, 256, 4, 2, 1),   # -> 256 x 8 x 8
            self._block(256, 512, 4, 2, 1),   # -> 512 x 4 x 4
            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=0), # -> 1 x 1 x 1
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, latent_dim, channels_img):
        super(Generator, self).__init__()
        # input: N x latent_dim x 1 x 1
        self.gen = nn.Sequential(
            self._block(latent_dim, 1024, 4, 1, 0), # -> 1024 x 4 x 4
            self._block(1024, 512, 4, 2, 1), # -> 512 x 8 x 8
            self._block(512, 256, 4, 2, 1),  # -> 256 x 16 x 16
            self._block(256, 128, 4, 2, 1),  # -> 128 x 32 x 32
            nn.ConvTranspose2d(128, channels_img, kernel_size=4, stride=2, padding=1), # -> channels_img x 64 x 64
            nn.Tanh(), # To scale output to [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)


generator = Generator(latent_dim, channels_img).to(device)
discriminator = Discriminator(channels_img).to(device)

criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]),
])

dataset = datasets.MNIST(root="dataset/", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

fixed_noise = torch.randn(64, latent_dim, 1, 1).to(device)

for epoch in range(n_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        batch_size = real.shape[0]

        ### train discriminator: max log(d(x)) + log(1 - d(g(z)))
        
        # clear old gradients from the discriminator
        discriminator.zero_grad()
        
        # part 1: train with real images
        # create labels for real images (all ones)
        label_real = torch.ones(batch_size, 1, 1, 1).to(device)
        # forward pass real images through discriminator
        output_real = discriminator(real)
        # calculate the loss for real images (how far the discriminator's predictions are from 1)
        loss_d_real = criterion(output_real, label_real)
        # calculate gradients for this part
        loss_d_real.backward()
        
        # part 2: train with fake images
        # generate a batch of random noise
        noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        # generate a batch of fake images from the noise
        fake = generator(noise)
        # create labels for fake images (all zeros)
        label_fake = torch.zeros(batch_size, 1, 1, 1).to(device)
        # forward pass fake images through discriminator
        # we use .detach() to stop gradients from flowing back to the generator during the discriminator's turn
        output_fake = discriminator(fake.detach())
        # calculate the loss for fake images (how far the discriminator's predictions are from 0)
        loss_d_fake = criterion(output_fake, label_fake)
        # calculate gradients for this part, accumulating them with the real image gradients
        loss_d_fake.backward()
        
        # update the discriminator's weights using the accumulated gradients
        optimizer_d.step()
        
        ### train generator: min log(1 - d(g(z))) <-> max log(d(g(z)))
        
        # clear old gradients from the generator
        generator.zero_grad()
        # forward pass the same fake images through the updated discriminator
        output_on_fake = discriminator(fake)
        # calculate the generator's loss. we use real_labels (ones) to trick the discriminator.
        # the generator wins if the discriminator thinks its fake images are real.
        loss_g = criterion(output_on_fake, label_real)
        # calculate gradients for the generator
        loss_g.backward()
        # update the generator's weights
        optimizer_g.step()

        # print losses and save generated images periodically
        if batch_idx % 100 == 0:
            print(
                f"epoch [{epoch}/{n_epochs}] batch {batch_idx}/{len(loader)} "
                f"loss d: {loss_d_real.item() + loss_d_fake.item():.4f}, loss g: {loss_g.item():.4f}"
            )
            # saving generated images for inspection
            with torch.no_grad():
                fake_samples = generator(fixed_noise)
                img_grid = torchvision.utils.make_grid(fake_samples, normalize=True)
                save_image(img_grid, f"generated_epoch_{epoch}.png")