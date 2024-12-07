Color Quantizer using PyTorch
Project Overview
This project implements a simple Color Quantizer using PyTorch. Color Quantization is the process of reducing the number of colors in an image, which can be useful for various applications such as image compression, artistic effects, or preparing images for devices with limited color display capabilities. This implementation utilizes a PyTorch neural network model to achieve color quantization through clustering, specifically employing the Mean Squared Error (MSE) as the loss function and Adam optimizer for training.

Key Features:

Dynamic Cluster Specification: Easily adjust the number of color clusters (quantization levels) to suit your application.
PyTorch Powered: Leveraging PyTorch for efficient model training and inference.
Visualization: Includes a simple visualization script to compare the original image with its quantized version.
Getting Started
Prerequisites
Python 3.x (Recommended: Latest version)
PyTorch (Installation instructions: https://pytorch.org/get-started/locally/)
** Torchvision** (Installed alongside PyTorch)
PIL (Python Imaging Library) (Installed alongside PyTorch, or separately if needed)
Matplotlib for visualization (Installation: pip install matplotlib)
Optional but Recommended: A compatible GPU for accelerated training ( MPS for Apple Silicon, CUDA for NVIDIA GPUs)
Installation
Clone this repository:
git clone https://github.com/YourGitHubUsername/ColorQuantizer.git

Navigate into the cloned directory:
cd ColorQuantizer

Ensure you have all prerequisites installed. If you've cloned the repo, all Python dependencies should be installable via:
pip install -r requirements.txt

Note: If you don't see a requirements.txt file, you can create one with the following content and then run the above command:
torch
torchvision
pillow
matplotlib

Running the Color Quantizer
Prepare Your Image: Place your desired image in the project root directory or update the image_path variable in the script to point to your image.
Execute the Script:
python color_quantizer.py

Customization: Adjust the num_clusters variable in color_quantizer.py to change the quantization level.
Example Use Case:

Input Image: A high-resolution photograph with a wide color gamut.
Output: A quantized image with a significantly reduced color palette, suitable for display on limited devices or for achieving a stylized effect.
Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

Fork the Project
Create your Feature Branch (git checkout -b feature/AmazingFeature)
Commit your Changes (git commit -m 'Add some AmazingFeature')
Push to the Branch (git push origin feature/AmazingFeature)
Open a Pull Request
License
Distributed under the MIT License. See LICENSE for more information.

Contact
Author: Akhil Kumar
GitHub: https://github.com/akhil4674
Acknowledgments
PyTorch and TorchVision teams for their incredible work.
PIL and Matplotlib developers for their contributions to the Python ecosystem.
