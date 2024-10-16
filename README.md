# Implementation of filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step 1:
Import necessary libraries: OpenCV, NumPy, and Matplotlib.Read an image, convert it to RGB format, define an 11x11 averaging kernel, and apply 2D convolution filtering.Display the original and filtered images side by side using Matplotlib.

### Step 2:
Define a weighted averaging kernel (kernel2) and apply 2D convolution filtering to the RGB image (image2).Display the resulting filtered image (image4) titled 'Weighted Averaging Filtered' using Matplotlib's imshow function.

### Step 3:

Apply Gaussian blur with a kernel size of 11x11 and standard deviation of 0 to the RGB image (image2).Display the resulting Gaussian-blurred image (gaussian_blur) titled 'Gaussian Blurring Filtered' using Matplotlib's imshow function.
### Step 4:
Apply median blur with a kernel size of 11x11 to the RGB image (image2).Display the resulting median-blurred image (median) titled 'Median Blurring Filtered' using Matplotlib's imshow function.

### Step 5 :
Define a Laplacian kernel (kernel3) and perform 2D convolution filtering on the RGB image (image2).Display the resulting filtered image (image5) titled 'Laplacian Kernel' using Matplotlib's imshow function.
### Step 6 :
Apply the Laplacian operator to the RGB image (image2) using OpenCV's cv2.Laplacian function.Display the resulting image (new_image) titled 'Laplacian Operator' using Matplotlib's imshow function.

## Program:
```
Developed By   : ASHWIN KUMAR S
Register No : 212222240013
```

### 1. Smoothing Filters

i) Averaging Filter
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image in grayscale
image = cv2.imread('kamal1jpg', cv2.IMREAD_GRAYSCALE)

# Add salt-and-pepper noise to the image
salt_prob = 0.05  # Probability of salt noise
pepper_prob = 0.05  # Probability of pepper noise

# Create a copy of the original image
noisy_image = np.copy(image)

# Salt noise (white pixels)
num_salt = np.ceil(salt_prob * image.size)
coords_salt = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
noisy_image[tuple(coords_salt)] = 255

# Pepper noise (black pixels)
num_pepper = np.ceil(pepper_prob * image.size)
coords_pepper = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
noisy_image[tuple(coords_pepper)] = 0

# Manually apply a 3x3 box filter
filtered_image = np.zeros_like(noisy_image)  # Create an empty output image

# Get image dimensions
height, width = noisy_image.shape

# Iterate through the image, skipping the border pixels
for i in range(1, height - 1):
    for j in range(1, width - 1):
        # Extract the 3x3 neighborhood
        neighborhood = noisy_image[i - 1:i + 2, j - 1:j + 2]
        
        # Compute the mean of the neighborhood
        filtered_value = np.mean(neighborhood)
        
        # Assign the filtered value to the output image
        filtered_image[i, j] = filtered_value

# Display the images side by side using Matplotlib
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image (Salt-and-Pepper)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image (Manual Box Filter 3x3)')
plt.axis('off')

plt.tight_layout()
plt.show()

```
ii) Weighted Averaging Filter
```

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image in grayscale
image = cv2.imread('kamal1jpg', cv2.IMREAD_GRAYSCALE)

# Add salt-and-pepper noise to the image
salt_prob = 0.05  # Probability of salt noise
pepper_prob = 0.05  # Probability of pepper noise

# Create a copy of the original image
noisy_image = np.copy(image)

# Salt noise (white pixels)
num_salt = np.ceil(salt_prob * image.size)
coords_salt = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
noisy_image[tuple(coords_salt)] = 255

# Pepper noise (black pixels)
num_pepper = np.ceil(pepper_prob * image.size)
coords_pepper = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
noisy_image[tuple(coords_pepper)] = 0

# Define a 3x3 weighted average kernel
kernel = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]]) / 16.0  # Normalize the kernel

# Get the dimensions of the image and kernel
image_height, image_width = noisy_image.shape
kernel_size = kernel.shape[0]  # Assuming square kernel
pad = kernel_size // 2

# Create a padded image to handle the borders
padded_image = np.pad(noisy_image, pad, mode='constant', constant_values=0)

# Create an output image to store the filtered result
filtered_image = np.zeros_like(noisy_image)

# Manually apply convolution
for i in range(pad, image_height + pad):
    for j in range(pad, image_width + pad):
        # Extract the region of interest (ROI)
        roi = padded_image[i - pad:i + pad + 1, j - pad:j + pad + 1]
        
        # Perform element-wise multiplication and sum the result
        filtered_value = np.sum(roi * kernel)
        
        # Assign the filtered value to the output image
        filtered_image[i - pad, j - pad] = np.clip(filtered_value, 0, 255)

# Display the images side by side using Matplotlib
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image (Salt-and-Pepper)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image (Manual Weighted Avg)')
plt.axis('off')

plt.tight_layout()
plt.show()

```
iii) Gaussian Filter
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image in grayscale
image = cv2.imread('kamal1jpg', cv2.IMREAD_GRAYSCALE)

# Add Gaussian noise to the image
mean = 0
sigma = 25
gaussian_noise = np.random.normal(mean, sigma, image.shape)
noisy_image = image + gaussian_noise
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# Manually apply a 3x3 box filter
filtered_image = np.zeros_like(noisy_image)  # Create an empty output image

# Get image dimensions
height, width = noisy_image.shape

# Iterate through the image, skipping the border pixels
for i in range(1, height - 1):
    for j in range(1, width - 1):
        # Extract the 3x3 neighborhood
        neighborhood = noisy_image[i - 1:i + 2, j - 1:j + 2]
        
        # Compute the mean of the neighborhood
        filtered_value = np.mean(neighborhood)
        
        # Assign the filtered value to the output image
        filtered_image[i, j] = filtered_value

# Display the images side by side using Matplotlib
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image (Gaussian Noise)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image (Manual Box Filter 3x3)')
plt.axis('off')

plt.tight_layout()
plt.show()

```
iv)Using Median Filter
```

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image in grayscale
image = cv2.imread('kamal1jpg', cv2.IMREAD_GRAYSCALE)

# Add salt-and-pepper noise to the image
salt_prob = 0.05  # Probability of salt noise
pepper_prob = 0.05  # Probability of pepper noise

# Create a copy of the original image
noisy_image = np.copy(image)

# Salt noise (white pixels)
num_salt = np.ceil(salt_prob * image.size)
coords_salt = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
noisy_image[tuple(coords_salt)] = 255

# Pepper noise (black pixels)
num_pepper = np.ceil(pepper_prob * image.size)
coords_pepper = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
noisy_image[tuple(coords_pepper)] = 0

# Manually apply a 3x3 median filter
filtered_image = np.zeros_like(noisy_image)  # Create an empty output image

# Get image dimensions
height, width = noisy_image.shape

# Iterate through the image, skipping the border pixels
for i in range(1, height - 1):
    for j in range(1, width - 1):
        # Extract the 3x3 neighborhood
        neighborhood = noisy_image[i - 1:i + 2, j - 1:j + 2]
        
        # Flatten the neighborhood and compute the median
        median_value = np.median(neighborhood)
        
        # Assign the median value to the output image
        filtered_image[i, j] = median_value

# Display the images side by side using Matplotlib
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image (Salt-and-Pepper)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image (Manual Median Filter)')
plt.axis('off')

plt.tight_layout()
plt.show()

```

### 2. Sharpening Filters
i) Using Laplacian Linear Kernal
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image in grayscale
image = cv2.imread('kamal1jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

# Define the Laplacian kernel
laplacian_kernel = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]])

# Get image and kernel dimensions
image_height, image_width = blurred_image.shape
kernel_height, kernel_width = laplacian_kernel.shape

# Calculate padding size
pad_height = kernel_height // 2
pad_width = kernel_width // 2

# Pad the image to handle borders
padded_image = np.pad(blurred_image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

# Create an empty image for the Laplacian filter output
laplacian_image = np.zeros_like(blurred_image)

# Apply the Laplacian filter manually
for i in range(image_height):
    for j in range(image_width):
        # Extract the region of interest
        region = padded_image[i:i + kernel_height, j:j + kernel_width]
        # Apply the kernel to the region
        laplacian_value = np.sum(region * laplacian_kernel)
        # Assign the result to the output image
        laplacian_image[i, j] = laplacian_value

# Normalize Laplacian image to be in the same range as the original image
laplacian_image = np.clip(laplacian_image, 0, 255).astype(np.uint8)

# Add the original image and the Laplacian image to get the final output
sharpened_image = cv2.add(image, laplacian_image)

# Display the images side by side using Matplotlib
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(laplacian_image, cmap='gray')
plt.title('Laplacian Filtered Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sharpened_image, cmap='gray')
plt.title('Sharpened Image')
plt.axis('off')

plt.tight_layout()
plt.show()

```
ii) Using Laplacian Operator
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image in grayscale
image = cv2.imread('kamal1jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

# Apply the Laplacian operator using OpenCV
laplacian_image = cv2.Laplacian(blurred_image, cv2.CV_64F, ksize=3)

# Take absolute values and normalize the Laplacian image to 8-bit
laplacian_image = np.absolute(laplacian_image)
laplacian_image = np.clip(laplacian_image, 0, 255).astype(np.uint8)

# Add the original image and the Laplacian image to get the final output
sharpened_image = cv2.add(image, laplacian_image)

# Display the images side by side using Matplotlib
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(laplacian_image, cmap='gray')
plt.title('Laplacian Filtered Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sharpened_image, cmap='gray')
plt.title('Sharpened Image')
plt.axis('off')

plt.tight_layout()
plt.show()

```

## OUTPUT:
### 1. Smoothing Filters


i) Averaging Filter

![image](https://github.com/user-attachments/assets/3a6c17b0-27cd-49b4-8b1b-74f52b135c97)

ii)Using Weighted Averaging Filter

![image](https://github.com/user-attachments/assets/4b0c7557-5d94-4ded-867c-371b6638475d)


iii)Using Gaussian Filter

![image](https://github.com/user-attachments/assets/17128445-96ae-4e95-b264-d02aa261ee8d)


iv) Using Median Filter

![image](https://github.com/user-attachments/assets/443aec85-a5a8-40b9-8213-b2aa5fc95fa5)


### 2. Sharpening Filters
i) Using Laplacian Kernal

![image](https://github.com/user-attachments/assets/175323da-6066-4686-b98a-a8c8e152a581)


ii) Using Laplacian Operator

![image](https://github.com/user-attachments/assets/4501d9da-0fa5-4a51-adac-948cd9ef4cb5)


## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
