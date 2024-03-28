from scipy.signal import convolve2d
import numpy as np
from skimage import data, color

def depthwise_convolution(image, kernel):
    output = np.zeros_like(image)
    for i in range(image.shape[2]):  
        output[:, :, i] = convolve2d(image[:, :, i], kernel, mode='same', boundary='wrap')
    return output

def pointwise_convolution(image, kernels):

    output = np.sum(image * kernels, axis=2)
    return output

# Example usage
image = color.rgb2gray(data.astronaut()) 
image = image[:, :, np.newaxis] 
kernel_depthwise = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  
kernel_pointwise = np.array([[[1]], [[0]], [[-1]]]) 


perform_depthwise = True
perform_pointwise = False

if perform_depthwise:
    convolved_image_depthwise = depthwise_convolution(image, kernel_depthwise)
    print("Completed Depthwise Convolution.")

if perform_pointwise:
    convolved_image_pointwise = pointwise_convolution(convolved_image_depthwise, kernel_pointwise)
    print("Completed Pointwise Convolution.")
