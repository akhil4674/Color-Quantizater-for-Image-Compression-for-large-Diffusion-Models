# Color Quantizer using PyTorch 🎨

## 📖 Project Overview

This project implements a **Color Quantizer** using **PyTorch**, which reduces the number of distinct colors in an image through clustering. Color Quantization is widely used for **image compression**, **artistic effects**, or optimizing images for devices with limited color display capabilities (e.g., old displays, low-color devices).

The color quantization process is achieved by clustering pixel colors and assigning them to a reduced set of color centers. This method leverages **K-means clustering** in the color space to minimize the **Mean Squared Error (MSE)** between original pixel values and the assigned color centers.

## 🧠 Key Concepts

### 1. **Color Quantization**
Color Quantization is the process of reducing the number of distinct colors in an image, aiming to represent the image with fewer colors. This is useful for applications such as:
- **Image compression**: Reducing file sizes while maintaining perceptual quality.
- **Artistic effects**: Achieving stylized looks, such as posterization or low-fidelity visuals.
- **Display adaptation**: Preparing images for devices with limited color palettes.

### 2. **Clustering for Color Quantization**
The core of the algorithm involves **clustering** pixel colors in the image into a smaller set of representative colors (color centers). The most common approach for this is **K-Means clustering**:
- **K-Means** groups pixels into `k` clusters, where each cluster represents a color center.
- Each pixel is then assigned to the nearest color center, effectively reducing the number of distinct colors.

### 3. **K-Means Clustering**
K-Means clustering is an iterative algorithm that partitions a dataset (in this case, pixel colors) into `k` clusters. The steps are:
1. **Initialization**: Choose `k` initial cluster centers (often randomly or by using K-means++ initialization).
2. **Assignment**: Assign each data point (pixel) to the nearest cluster center.
3. **Update**: Recalculate the cluster centers as the mean of the assigned points.
4. **Convergence**: Repeat steps 2 and 3 until the cluster centers no longer change significantly.

### 4. **Mean Squared Error (MSE) Loss**
During the quantization process, we minimize the **Mean Squared Error (MSE)** between the original pixel color and the assigned color center. The MSE loss function for a single pixel is defined as:

$$
L_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (x_i - c_i)^2
$$

Where:
- \( x_i \) is the original pixel color value.
- \( c_i \) is the color center to which the pixel is assigned.
- \( N \) is the number of pixels in the image.

The goal of the quantizer is to find the set of color centers that minimizes this loss over all pixels, resulting in a reduced color palette.

## 🔧 Key Functions and Components

### 1. **Color Quantization (Main Function)**
The main function of the quantizer processes an input image and performs the following tasks:
- Convert the image to a **color space** that makes clustering more effective (e.g., RGB or Lab color space).
- Flatten the image into a 2D array where each row represents a pixel and each column represents a color channel (e.g., R, G, B).
- Apply **K-means clustering** to group the pixels into a specified number of clusters (`num_clusters`).
- Replace the original pixel values with the closest cluster center, thereby reducing the number of colors in the image.

### 2. **K-Means Clustering Implementation**
In the K-means algorithm used in this quantizer:
- **Color Centers (Centroids)**: The centers of the clusters are calculated as the mean of all the pixels assigned to each cluster.
- **Pixel Assignment**: Each pixel is assigned to the nearest color center based on Euclidean distance in the RGB color space.
- **Update Step**: The algorithm updates the color centers after each iteration to minimize the MSE loss.

The K-means clustering process typically includes the following steps:
- Randomly initialize `k` centroids (color centers).
- For each pixel in the image, calculate the **Euclidean distance** from each centroid and assign the pixel to the closest one.
- After assigning all pixels, recalculate the centroids as the average of the assigned pixels.
- Repeat the assignment and update steps until convergence.

### 3. **Visualization**
Once the quantization process is complete, the quantized image is compared with the original image to visualize the reduction in the color palette. The visualization function can display both images side by side for comparison.

## 🔢 Key Formulas

### K-Means Objective Function
The K-means algorithm seeks to minimize the **within-cluster sum of squared distances** (WSS) between each pixel and its corresponding cluster center. The objective function is:

$$
\mathcal{L} = \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - c_k \|^2
$$

Where:
- \( K \) is the number of clusters (quantization levels).
- \( C_k \) is the set of pixels assigned to the k-th cluster.
- \( x_i \) is a pixel in the image.
- \( c_k \) is the color center (centroid) for the k-th cluster.

The algorithm minimizes this objective by updating the centroids until convergence.

### Euclidean Distance
The Euclidean distance is used to measure the proximity between a pixel and a color center. It is computed as:

$$
\text{distance}(x_i, c_k) = \sqrt{(x_{iR} - c_{kR})^2 + (x_{iG} - c_{kG})^2 + (x_{iB} - c_{kB})^2}
$$

Where:
- \( x_{iR}, x_{iG}, x_{iB} \) are the RGB components of the pixel \( x_i \).
- \( c_{kR}, c_{kG}, c_{kB} \) are the RGB components of the color center \( c_k \).

This distance is used to determine the closest cluster center for each pixel during the assignment step in K-means clustering.

---

## 🚀 Conclusion

This project leverages **K-means clustering** and **PyTorch** to efficiently perform **color quantization**, making it possible to reduce the number of colors in an image while retaining its general structure and visual appearance. The use of clustering helps in minimizing the **Mean Squared Error (MSE)** between original and quantized pixels, achieving an effective color reduction for various applications such as compression and stylization.

Let me know if you need further clarification or additional explanations on any part of the algorithm!
