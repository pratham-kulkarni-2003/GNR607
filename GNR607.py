import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def select_file():
    Tk().withdraw()
    file_path = askopenfilename(
        title="Select a File",
        filetypes=[
            ("All Files", "*.*"),  
            ("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif")  
        ]
    )
    return file_path


image_path = select_file()
if not image_path:
    print("No file selected. Exiting.")
    exit()


image = cv2.imread(image_path)
if image is None:
    print("Error: Unable to load the image. Ensure the selected file is an image.")
    exit()


height, width = image.shape[:2]
image = cv2.resize(image, (width // 4, height // 4))


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)


def apply_filter(image, kernel):
    height, width = image.shape
    output = np.zeros_like(image, dtype=np.float32)
    

    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    
 
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            region = padded_image[i-1:i+2, j-1:j+2]
            output[i-1, j-1] = np.sum(region * kernel)
    
    return output

grad_x = apply_filter(gray_image, sobel_x)
grad_y = apply_filter(gray_image, sobel_y)

# Compute the gradient magnitude
gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

# Normalize the gradient magnitude
gradient_magnitude -= gradient_magnitude.min()
gradient_magnitude /= gradient_magnitude.max()

# Ask the user to input a threshold value
threshold = float(input("Enter the threshold value (0-1): "))
if threshold < 0 or threshold > 1:
    print("Threshold value must be between 0 and 1.")
    exit()

# Apply thresholding
edge_image = (gradient_magnitude > threshold).astype(np.float32)

# Display the results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('Input Grayscale Image')
axes[0].axis('off')

axes[1].imshow(edge_image, cmap='gray')
axes[1].set_title('Edge Image (Thresholded)')
axes[1].axis('off')

plt.tight_layout()
plt.show()
