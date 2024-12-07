import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.transforms import Resize
from PIL import Image
import matplotlib.pyplot as plt
import time

class ColorQuantizer(nn.Module):
    def __init__(self, num_clusters):
        super(ColorQuantizer, self).__init__()
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, 3))

    def forward(self, x):
        x_expanded = x.unsqueeze(1)
        centers_expanded = self.cluster_centers.unsqueeze(0)
        distances = torch.norm(x_expanded - centers_expanded, dim=2)
        _, indices = distances.min(1)
        return self.cluster_centers[indices]

def run_quantizer(device):
    # Assuming the image path is correctly set as provided
    image_path = "/Users/akhilkumar/Desktop/vic_encoding/porsche.png"
    original_image = Image.open(image_path).convert('RGB')
    resize_transform = Resize((128, 128))
    resized_image = resize_transform(original_image)
    image_tensor = to_tensor(resized_image)

    # Flatten image to a list of RGB colors
    input_data = image_tensor.view(-1, 3).to(device)

    # Model and optimizer
    num_clusters = 4
    model = ColorQuantizer(num_clusters).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    times = []
    num_epochs = 1000  # Adjust the number of epochs as necessary
    for epoch in range(num_epochs):
        start_time = time.time()
        output = model(input_data)
        loss = criterion(output, input_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
        print(f'Epoch {epoch}, Loss: {loss.item()}, Time: {end_time - start_time} sec')

    return times

# Set devices for comparison
cpu_device = torch.device("cpu")
gpu_device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Run experiments
cpu_times = run_quantizer(cpu_device)
gpu_times = run_quantizer(gpu_device) if torch.backends.mps.is_available() else cpu_times

# Plotting
epochs = range(1, len(cpu_times) + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs, cpu_times, label='CPU', color='red')
plt.plot(epochs, gpu_times, label='M1 GPU (MPS)', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Time in Seconds')
plt.title('Training Time Comparison: CPU vs Apple M1 GPU (MPS)')
plt.legend()
plt.show()