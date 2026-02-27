import matplotlib
from scipy.io import loadmat
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
# import napari

# Load the .mat file
file_name = "Spectrum-Simplified.mat"
file_path = loadmat(r"E:/Liam/HSI-3DCNN-Project/hsi_datasets/v303/" + file_name)
cube = file_path['DataCube']
save_path = r"E:/Liam/HSI-3DCNN-Project/evaluation_results/"
print(f"Shape: {cube.shape}")
print(f"Dtype: {cube.dtype}")
print(f"Range: [{cube.min()}, {cube.max()}]")

# # Spectral signature at one pixel
# plt.plot(cube[45, 100, :])  # Assuming (Y, X, B)
# plt.title("Spectral signature")
# plt.show()

plt.plot(cube[45, 100, :])
plt.savefig(save_path + 'spectral_signature.png')
plt.close()

# cube = data['DataCube']
cube = np.transpose(cube, (1,2,0))  # Fix in memory
print(f"New shape: {cube.shape}")  # Should be (1632, 1232, 110)
plt.imshow(cube[:, :, 10], cmap='gray')

# # Spatial image at one band
# plt.imshow(cube[:, :, 10], cmap='gray')
# plt.title("Band 10")
# plt.show()

plt.imshow(cube[:, :, 10], cmap='gray')
plt.savefig(save_path + 'band_10.png')
plt.close()