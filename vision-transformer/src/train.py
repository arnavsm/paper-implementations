import os
import torch
import torchvision
from torchvision import transforms
from pathlib import Path
import json
import matplotlib.pyplot as plt

import data_setup, engine, model_builder, utils

# Define paths
data_path = Path("./data/")
train_dir = data_path / "CIFAR10/train"
test_dir = data_path / "CIFAR10/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 32
HIDDEN_UNITS = 32
LEARNING_RATE = 0.003

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=str(train_dir),
    test_dir=str(test_dir),
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Initialize the model
model = model_builder.VisionTransformer(
    image_size=64,
    patch_size=16,
    num_classes=len(class_names),
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
    dropout_rate=0.1
)

# Setup optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

# Directory for saving checkpoints and logs
checkpoint_dir = "../vision-transformer/models"
log_dir = "../vision-transformer/logs"
plots_dir = "../vision-transformer/docs/images/plots"

Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
Path(log_dir).mkdir(parents=True, exist_ok=True)
Path(plots_dir).mkdir(parents=True, exist_ok=True)

# Train the model
results = engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
    device=device,
    checkpoint_dir=checkpoint_dir
)

# Save results
results_path = Path(log_dir) / "training_results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Final Results saved to: {results_path}")

# Print final results
print("Final Results:")
print(f"Train Loss: {results['train_loss'][-1]:.4f}")
print(f"Train Accuracy: {results['train_acc'][-1]:.4f}")
print(f"Test Loss: {results['test_loss'][-1]:.4f}")
print(f"Test Accuracy: {results['test_acc'][-1]:.4f}")

# Plot training and testing metrics
epochs = list(range(1, NUM_EPOCHS + 1))

plt.figure(figsize=(12, 5))

# Plot training and test loss
plt.subplot(1, 2, 1)
plt.plot(epochs, results['train_loss'], label='Train Loss')
plt.plot(epochs, results['test_loss'], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Test Loss')
plt.legend()

# Plot training and test accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, results['train_acc'], label='Train Accuracy')
plt.plot(epochs, results['test_acc'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy')
plt.legend()

# Save and show plots
plt.tight_layout()
plot_path = Path(plots_dir) / "training_plots.png"
plt.savefig(plot_path)
plt.show()
print(f"Training plots saved to: {plot_path}")
