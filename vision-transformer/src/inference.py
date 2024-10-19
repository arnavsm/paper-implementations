import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import model_builder

class_names = []

# Define paths
checkpoint_path = "vision-transformer/models/best_model.pth"
class_names_path = "vision-transformer/logs/class_names.json"

# Load class names
def load_class_names(class_names_path):
    import json
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    return class_names

# Load the model
def load_model(checkpoint_path, device):
    model = model_builder.VisionTransformer(
        image_size=64,
        patch_size=16,
        num_classes=len(class_names),  # Load this from a file or define appropriately
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        dropout_rate=0.1
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Download and preprocess the image
def preprocess_image(image_url, transform):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = transform(img)
    return img.unsqueeze(0)  # Add batch dimension

# Perform inference
def infer(model, img, device, class_names):
    img = img.to(device)
    with torch.no_grad():
        pred = model(img)
    pred_class = torch.argmax(pred, dim=1).item()
    return class_names[pred_class]

# Main function
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and class names
    model = load_model(checkpoint_path, device)
    class_names = load_class_names(class_names_path)
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Example image URL (replace with any image URL you want to test)
    image_url = "https://example.com/random-image.jpg"
    
    # Preprocess the image
    img = preprocess_image(image_url, transform)
    
    # Perform inference
    predicted_class = infer(model, img, device, class_names)
    
    # Output results
    print(f"Predicted class: {predicted_class}")
    
    # Optionally, display the image
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    plt.imshow(img)
    plt.title(f"Predicted class: {predicted_class}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
