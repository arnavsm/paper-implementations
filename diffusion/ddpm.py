# diffusion on mnist (ddpm-style): predict noise with a tiny u-net and sample images.
# references (conceptually): ho et al., "denoising diffusion probabilistic models" (2020)

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# ---------------- model: a tiny u-net-ish backbone ----------------

class SinusoidalTimestepEmbedding(nn.Module):
    """
    creates sinusoidal positional embeddings for the timestep.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (batch,) tensor of integers in [0, T-1]
        # create sin/cos positional embeddings as described in "attention is all you need".
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t.float()[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # pad for odd dimensions
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
            
        return embeddings

class ResBlock(nn.Module):
    """
    a residual block with a time embedding projection.
    """
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, out_channels)
        )
        
        # skip connection if input and output channels differ
        self.skip_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_embedding):
        h = self.conv1(F.silu(self.norm1(x)))
        
        # add time embedding
        time_bias = self.time_mlp(time_embedding).unsqueeze(-1).unsqueeze(-1)
        h = h + time_bias
        
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip_connection(x)

class TinyUNet(nn.Module):
    """
    a small u-net model to predict the noise in an image.
    """
    def __init__(self, channels=64, time_embedding_dim=128):
        super().__init__()
        
        # timestep embedding projection
        self.time_embedding = nn.Sequential(
            SinusoidalTimestepEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )
        
        # u-net architecture
        self.in_conv = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        
        # down-sampling path
        self.down1 = ResBlock(channels, channels, time_embedding_dim)
        self.pool = nn.AvgPool2d(2) # 28x28 -> 14x14
        self.down2 = ResBlock(channels, channels * 2, time_embedding_dim)
        
        # middle block
        self.mid = ResBlock(channels * 2, channels * 2, time_embedding_dim)
        
        # up-sampling path
        self.up = nn.Upsample(scale_factor=2, mode="nearest") # 14x14 -> 28x28
        self.up1 = ResBlock(channels * 2, channels, time_embedding_dim)
        
        # final output convolution
        self.out_conv = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

    def forward(self, x, t):
        # embed the timestep
        time_emb = self.time_embedding(t)
        
        # initial convolution
        x = self.in_conv(x)
        
        # down-sampling
        down1_out = self.down1(x, time_emb)
        down2_in = self.pool(down1_out)
        down2_out = self.down2(down2_in, time_emb)
        
        # middle
        mid_out = self.mid(down2_out, time_emb)
        
        # up-sampling
        up1_in = self.up(mid_out)
        up1_out = self.up1(up1_in, time_emb)
        
        # add skip connection from the down-sampling path
        h = up1_out + down1_out
        
        # predict the noise
        predicted_noise = self.out_conv(F.silu(h))
        return predicted_noise

# ---------------- noise schedule utilities ----------------

class BetaSchedule:
    """
    manages the noise schedule (betas, alphas) for the diffusion process.
    """
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        # a linear schedule is used as a simple default.
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

# ---------------- forward noising (q) and training loss ----------------

def q_sample(x0, t, alpha_bars):
    """
    sample x_t from x_0 in a closed form using the reparameterization trick.
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    """
    batch_size = x0.shape[0]
    alpha_bar_t = alpha_bars[t].view(batch_size, 1, 1, 1)
    noise = torch.randn_like(x0)
    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise
    return xt, noise

# ---------------- reverse step (p) for sampling ----------------

def p_sample_step(model, xt, t, schedule):
    """
    performs one reverse step from x_t to x_{t-1} using the predicted noise.
    """
    betas, alphas, alpha_bars = schedule.betas, schedule.alphas, schedule.alpha_bars
    
    # get schedule values for the current timestep t
    beta_t = betas[t].view(-1, 1, 1, 1)
    alpha_t = alphas[t].view(-1, 1, 1, 1)
    alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
    
    # predict noise using the model
    predicted_noise = model(xt, t)
    
    # calculate the mean of the reverse distribution
    mean = (1.0 / torch.sqrt(alpha_t)) * (xt - beta_t / torch.sqrt(1.0 - alpha_bar_t) * predicted_noise)
    
    # add noise for all steps except the last one
    if t[0] > 0:
        z = torch.randn_like(xt)
    else:
        z = torch.zeros_like(xt)
        
    variance = beta_t
    return mean + torch.sqrt(variance) * z

@torch.no_grad()
def generate_samples(model, schedule, n=64, device="cpu"):
    """
    generates a batch of samples by running the full reverse diffusion process.
    """
    model.eval()
    # start with pure noise
    x = torch.randn(n, 1, 28, 28, device=device)
    
    # iterate backwards through the timesteps
    for step in reversed(range(schedule.T)):
        t = torch.full((n,), step, device=device, dtype=torch.long)
        x = p_sample_step(model, x, t, schedule)
        
    # clamp and rescale to [0, 1] for visualization
    x = x.clamp(-1, 1)
    x = (x + 1) / 2.0
    return x


def train_model(epochs=5, batch_size=128, lr=2e-4, T=300, device=None):
    """
    main training loop for the ddpm model on mnist.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs("ddpm_samples", exist_ok=True)

    # data transformation: scale images to [-1, 1] for stability
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0)
    ])
    
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # initialize model, optimizer, and noise schedule
    model = TinyUNet(channels=64, time_embedding_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    schedule = BetaSchedule(T=T, device=device)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        num_samples = 0
        
        for x0, _ in data_loader:
            x0 = x0.to(device)
            b = x0.shape[0]
            
            # sample a random timestep for each image in the batch
            t = torch.randint(0, T, (b,), device=device)
            
            # generate noised image and the noise that was added
            xt, added_noise = q_sample(x0, t, schedule.alpha_bars)
            
            # predict the noise
            predicted_noise = model(xt, t)
            
            # calculate mse loss between added and predicted noise
            loss = F.mse_loss(predicted_noise, added_noise)
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * b
            num_samples += b
            
        print(f"Epoch {epoch}: Train MSE {total_loss / num_samples:.4f}")

        # generate and save a grid of samples after each epoch
        with torch.no_grad():
            images = generate_samples(model, schedule, n=64, device=device).cpu()
            utils.save_image(images, f"ddpm_samples/epoch_{epoch:02d}.png", nrow=8)

    return model, schedule

if __name__ == "__main__":
    trained_model, training_schedule = train_model(epochs=5, T=300)
    
    # generate a final, larger grid of samples
    print("Training complete. Generating final sample grid...")
    with torch.no_grad():
        final_images = generate_samples(trained_model, training_schedule, n=100)
        utils.save_image(final_images, "ddpm_samples/final.png", nrow=10)
    print("Final samples saved to ddpm_samples/final.png")
