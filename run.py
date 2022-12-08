from utils.helpers import transform_image_to_kspace, load_dicom_image
import matplotlib.pyplot as plt
import numpy as np

loaded = load_dicom_image(r'/Users/sdas/Desktop/12th Grade/Quantum Lab/Code/data/Image-109.dcm')
print(loaded.shape)
# plt.imshow(loaded, cmap='gray')
# plt.show()

kspace = transform_image_to_kspace(loaded)
print(kspace.shape)
print(f'Min kspace val = {np.min(kspace)}\nMax kspace val = {np.max(kspace)}')


