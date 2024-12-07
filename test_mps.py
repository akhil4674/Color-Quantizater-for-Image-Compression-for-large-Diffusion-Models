import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.transforms import Resize
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load and resize image
image_path = "/Users/akhilkumar/Desktop/vic_encoding/porsche.png"
original_image = Image.open(image_path).convert('RGB')
resize_transform = Resize((128, 128))  # Resize for faster processing
resized_image = resize_transform(original_image)
image_tensor = to_tensor(resized_image).unsqueeze(0)

# Flatten image to a list of RGB colors
input_data = image_tensor.view(-1, 3)

# Move data to GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
input_data = input_data.to(device)

# Initialize cluster centers (randomly pick 16 colors from the image)
num_clusters = 16
indices = torch.randperm(input_data.size(0))[:num_clusters]
initial_centers = input_data[indices].clone().detach()
centers = initial_centers.to(device)
centers.requires_grad = False

def k_means(input_data, centers, num_iterations=100):
    for _ in range(num_iterations):
        # Compute distances from the data points to the cluster centers
        distances = torch.cdist(input_data, centers)
        
        # Assign each data point to the nearest cluster
        _, indices = distances.min(dim=1)
        
        # Update cluster centers
        for i in range(num_clusters):
            selected = input_data[indices == i]
            if len(selected) > 0:
                centers[i] = selected.mean(dim=0)
    
    return centers, indices

# Run K-means
centers, assignments = k_means(input_data, centers)

# Reconstruct the image from the quantized colors
quantized = centers[assignments].view(image_tensor.shape)
quantized_image = to_pil_image(quantized.cpu())

# Display the original and quantized images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(resized_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(quantized_image)
plt.title('Quantized Image')
plt.axis('off')
plt.show()
