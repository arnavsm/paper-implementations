import os
from pathlib import Path
from torchvision import datasets, transforms

# Define paths
data_path = Path("./data/")
train_dir = data_path / "CIFAR10/train"
test_dir = data_path / "CIFAR10/test"

# Ensure directories exist
train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Download CIFAR-10 dataset
trainset = datasets.CIFAR10(root=train_dir, train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root=test_dir, train=False, download=True, transform=transform)

print("CIFAR-10 dataset downloaded and ready for use.")


