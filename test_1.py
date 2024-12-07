import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def load_image(image_path):
    # Load an image and transform to tensor
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def color_quantization(image_tensor, num_colors):
    orig_shape = image_tensor.shape
    image_flattened = image_tensor.view(-1, 3).to(device)

    # Ensure dtype is float32 for all computations
    image_flattened = image_flattened.type(torch.float32)

    # Randomly initialize colors
    colors = torch.rand(num_colors, 3, device=device, dtype=torch.float32)

    # Iteratively refine the colors
    for _ in range(100):  # Number of iterations
        closest = torch.zeros(image_flattened.shape[0], dtype=torch.long, device=device)
        
        # Manually compute distances to avoid torch.cdist
        for i, color in enumerate(colors):
            # Calculate the squared distance to each color
            distances = (image_flattened - color).pow(2).sum(1)
            if i == 0:
                min_distances = distances
            else:
                # Find the closest color
                mask = distances < min_distances
                closest[mask] = i
                min_distances[mask] = distances[mask]

        # Update each color to the mean of the pixels assigned to it
        for i in range(num_colors):
            mask = (closest == i)
            if mask.any():
                colors[i] = image_flattened[mask].mean(dim=0)

    # Recreate image
    new_image_flattened = colors[closest]
    new_image = new_image_flattened.view(orig_shape)
    return new_image

# Assuming you are using an Apple M1, set device to GPU if available
device = torch.device("mps")  # Apple M1 GPU device
image_path = '/Users/akhilkumar/Desktop/vic_encoding/porsche.png'
num_colors = 2  # Number of colors for quantization

image_tensor = load_image(image_path)
image_tensor = image_tensor.to(device)

quantized_image_tensor = color_quantization(image_tensor, num_colors)
quantized_image = transforms.ToPILImage()(quantized_image_tensor.squeeze(0).cpu())

quantized_image.save('quantized_image.jpg')
quantized_image.show()