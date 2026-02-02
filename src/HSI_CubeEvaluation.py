from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

# Load the .mat file
file_name = "Spectrum-Simplified.mat"
file_path = loadmat(r"E:/Liam's Projects/HSI-3DCNN-Project/hsi_datasets/v303/" + file_name)
cube = file_path['DataCube']
print(f"Shape: {cube.shape}")
print(f"Dtype: {cube.dtype}")
print(f"Range: [{cube.min()}, {cube.max()}]")

# Spectral signature at one pixel
plt.plot(cube[45, 100, :])  # Assuming (Y, X, B)
plt.title("Spectral signature")
plt.show()

# cube = data['DataCube']
cube = np.transpose(cube, (1,2,0))  # Fix in memory
print(f"New shape: {cube.shape}")  # Should be (1632, 1232, 110)
plt.imshow(cube[:, :, 10], cmap='gray')

# Spatial image at one band
plt.imshow(cube[:, :, 10], cmap='gray')
plt.title("Band 10")
plt.show()