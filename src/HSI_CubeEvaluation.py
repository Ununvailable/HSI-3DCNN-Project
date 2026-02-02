from scipy.io import loadmat
data = loadmat(r"E:/Liam's Projects/HSI-3DCNN-Project/hsi_datasets/v303/Spectrum-Simplified.mat")
cube = data['DataCube']
print(f"Shape: {cube.shape}")
print(f"Dtype: {cube.dtype}")
print(f"Range: [{cube.min()}, {cube.max()}]")

# Spectral signature at one pixel
plt.plot(cube[45, 100, :])  # Assuming (Y, X, B)
plt.title("Spectral signature")
plt.show()

# Spatial image at one band
plt.imshow(cube[:, :, 10], cmap='gray')
plt.title("Band 10")
plt.show()