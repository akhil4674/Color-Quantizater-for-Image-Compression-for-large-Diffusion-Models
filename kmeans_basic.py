import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from skimage import io

def color_quantization(image_path, k=4):
    # Load image
    image = io.imread(image_path)
    image = image / 255.0  # Normalize the image
    rows, cols, dims = image.shape
    
    # Flatten the image
    image_flattened = image.reshape(rows * cols, dims)
    
    # Apply k-means
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(image_flattened)
    quantized_colors = kmeans.cluster_centers_
    
    # Map each pixel to the centroid of its cluster
    quantized_image = quantized_colors[kmeans.labels_].reshape(rows, cols, dims)
    
    return quantized_image, kmeans, image_flattened

def plot_elbow_method(X, max_k=4):
    wcss = []
    for i in range(1, max_k+1):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_k+1), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-cluster Sum of Squares')
    plt.show()

def plot_kmeans_clusters(X, kmeans):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=kmeans.labels_.astype(float), s=50)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], 
               s=200, c='white', marker='*', edgecolor='black')
    ax.set_title('3D plot of KMeans Clusters')
    plt.show()


# Usage
image_path = '/Users/akhilkumar/Desktop/vic_encoding/porsche.png' # Replace with your image path
k = 8 # You may adjust this after running the elbow method

quantized_image, kmeans, image_flattened = color_quantization(image_path, k=k)
plot_elbow_method(image_flattened, max_k=10)
plot_kmeans_clusters(image_flattened, kmeans)

plt.imshow(quantized_image)
plt.axis('off')
plt.show()
