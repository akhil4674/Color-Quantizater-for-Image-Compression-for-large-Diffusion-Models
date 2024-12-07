import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.transforms import Resize
from PIL import Image
import matplotlib.pyplot as plt

class ColorQuantizer(nn.Module):
    def __init__(self, num_clusters):
        super(ColorQuantizer, self).__init__()
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, 3)) # 3D vector 

    def forward(self, x):
        # Expand input and cluster_centers to compute distance
        x_expanded = x.unsqueeze(1)  # Shape: (N, 1, 3)
        centers_expanded = self.cluster_centers.unsqueeze(0)  # Shape: (1, num_clusters, 3)
        distances = torch.norm(x_expanded - centers_expanded, dim=2)  # Euclidean distance
        _, indices = distances.min(1)
        return self.cluster_centers[indices]

# Load and preprocess image
image_path = "/Users/akhilkumar/Desktop/vic_encoding/porsche.png"
original_image = Image.open(image_path).convert('RGB')
resize_transform = Resize((256, 256))
resized_image = resize_transform(original_image)
image_tensor = to_tensor(resized_image).unsqueeze(0) #again shape changing

# Flatten image to a list of RGB colors
input_data = image_tensor.view(-1, 3)

# Move data to GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
input_data = input_data.to(device)

# Model and optimizer
num_clusters = 128  # Change number of clusters/colors here
model = ColorQuantizer(num_clusters).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01) # using adam optimizer for the GRAGIENT descent This value means that each step of the weight update is relatively smal
criterion = nn.MSELoss()

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    output = model(input_data)
    loss = criterion(output, input_data)
    optimizer.zero_grad()
    loss.backward()  # back propagation 
    optimizer.step()
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# After training, create quantized image
output = model(input_data).view(image_tensor.shape)
reconstructed_image = to_pil_image(output.squeeze().cpu())

# Show images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(original_image)  # Use original image for display
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image)
plt.title('Quantized Image')
plt.axis('off')
plt.show()