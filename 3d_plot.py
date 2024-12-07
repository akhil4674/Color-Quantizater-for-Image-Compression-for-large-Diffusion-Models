import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_image(image_path):
    return Image.open(image_path)

def extract_colors(image, num_colors=16):
    data = np.array(image)
    reshaped_data = data.reshape((-1, 3))
    model = KMeans(n_clusters=num_colors)
    labels = model.fit_predict(reshaped_data)
    palette = model.cluster_centers_
    return reshaped_data, palette

def plot_color_distribution(colors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(colors[:, 0], colors[:, 1], colors[:, 2], c=colors/255.0)
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    plt.title('3D Color Space')
    plt.show()

def plot_color_palette(palette):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.indices((2, 2, 2))
    x = x.flatten() - 0.5
    y = y.flatten() - 0.5
    z = z.flatten() - 0.5
    ax.bar3d(x, y, z, 1, 1, 1, color=palette/255.0, shade=True)
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    plt.title('Color Palette Cube')
    plt.show()

image_path = '/Users/akhilkumar/Desktop/vic_encoding/porsche.png'
image = load_image(image_path)
colors, palette = extract_colors(image)

plot_color_distribution(colors)
plot_color_palette(palette)
