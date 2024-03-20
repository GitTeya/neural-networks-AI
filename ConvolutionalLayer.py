import numpy as np

def convolve2d(image, kernel):
    #Without padding and striding
    
    # Get the spatial dimensions of the image and the kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Determine the size of the output image
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    
    # Initialize the output image with zeros
    output = np.zeros((output_height, output_width))
    
    # Slide the kernel over the image
    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)
            
    return output

# Example
image = np.array([
    [1, 2, 3, 4, 5, 6],
    [7, 8, 9, 10, 11, 12],
    [13, 14, 15, 16, 17, 18],
    [19, 20, 21, 22, 23, 24],
    [25, 26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35, 36]
])

kernel = np.array([
    [2, 0, 1],
    [0, 1, 2],
    [1, 0, 2]
])

# Perform convolution
convolved_image = convolve2d(image, kernel)
print(convolved_image)
