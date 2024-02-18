import numpy as np
from scipy.ndimage import convolve1d

# Use the following kernels for computing the image gradients
# G: a Gaussian smoothing kernel, used to smooth across the axes orthogonal to the gradient AFTER the gradient is computed
# H: the derivative kernel, that outputs a measure of the difference between neighboring pixels (derivative of a Gaussian)
KERNEL_G = np.array([0.015625, 0.093750, 0.234375, 0.312500, 0.234375, 0.093750, 0.015625])
KERNEL_H = np.array([0.03125, 0.12500, 0.15625, 0, -0.15625, -0.1250, -0.03125])

def compute_Ix(imgs):
    """
    Compute the gradient of the images along the x-dimension.
    
    WARNING: The first coordinate of the gradient is actually the y-coordinate,
    which means here you're computing the gradient along axis=1, NOT 0!
    
    Inputs:
        - imgs: image volume, axis semantics: first = y, second = x, third = t - shape: (H, W, N)
    Outputs:
        - Ix: image gradient along the x-dimension - shape: (H, W, N)
    """
    
    ### STUDENT CODE START ###
    Ix = np.zeros(imgs.shape) # shape: (H, W, N)
    for i in range(imgs.shape[2]): # iterate over all images
        Ix = convolve1d(imgs[:,:], KERNEL_H, axis=1) # compute the gradient along the x-dimension
        Ix = convolve1d(Ix[:,:], KERNEL_G, axis=0) # smooth the gradient along the y-dimension
        Ix = convolve1d(Ix[:,:], KERNEL_G, axis=2) # smooth the gradient along the t-dimension
    
    
    ### STUDENT CODE END ###
    
    return Ix

def compute_Iy(imgs):
    """
    Compute the gradient of the images along the x-dimension.
    
    WARNING: The first coordinate of the gradient is actually the y-coordinate,
    which means here you're computing the gradient along axis=0, NOT 1!
    
    Inputs:
        - imgs: image volume, axis semantics: first = y, second = x, third = t - shape: (H, W, N)
    Outputs:
        - Iy: image gradient along the y-dimension - shape: (H, W, N)
    """
    
    ### STUDENT CODE START ###
    Iy = np.zeros(imgs.shape) # shape: (H, W, N)
    for i in range(imgs.shape[2]): # iterate over all images
        Iy = convolve1d(imgs[:,:],KERNEL_H, axis=0) # compute the gradient along the y-dimension
        Iy = convolve1d(Iy[:,:], KERNEL_G, axis=1) # smooth the gradient along the x-dimension
        Iy = convolve1d(Iy[:,:], KERNEL_G, axis=2) # smooth the gradient along the t-dimension
        
        
    ### STUDENT CODE END ###
    
    return Iy

def compute_It(imgs):
    """
    Compute the gradient of the images along the x-dimension.
    
    WARNING: The first coordinate of the gradient is actually the y-coordinate,
    which means here you're computing the gradient along axis=0, NOT 1!
    
    Inputs:
        - imgs: image volume, axis semantics: first = y, second = x, third = t - shape: (H, W, N)
    Outputs:
        - It: temporal image gradient - shape: (H, W, N)
    """
    
    ### STUDENT CODE START ###
    It = np.zeros(imgs.shape) # shape: (H, W, N)
    for i in range(imgs.shape[2]): # iterate over all images except the last one
        It = convolve1d(imgs[:,:],KERNEL_H, axis=2) # compute the gradient along the t-dimension
        It = convolve1d(It[:,:], KERNEL_G, axis=0) # smooth the gradient along the y-dimension
        It = convolve1d(It[:,:], KERNEL_G, axis=1)
       ### STUDENT CODE END ###
    
    return It
