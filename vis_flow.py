import numpy as np
import matplotlib.pyplot as plt

def plot_flow(image, flow_image, confidence, threshmin=10):
    """
    Plot a flow field of one frame of the data.
    
    Inputs:
        - image: grayscale image - shape: (H, W)
        - flow_image: optical flow - shape: (H, W, 2)
        - confidence: confidence of the flow estimates - shape: (H, W)
        - threshmin: threshold for confidence (optional) - scalar
    """
    
    ### STUDENT CODE START ###
    
    # Useful function: np.meshgrid()
    # Hint: Use plt.imshow(<your image>, cmap='gray') to display the image in grayscale
    # Hint: Use plt.quiver(..., color='red') to plot the flow field on top of the image in a visible manner
    
    

    # generate a meshgrid for the image
    a = np.arange(0, image.shape[1], 1)
    b = np.arange(0, image.shape[0], 1)
    x, y = np.meshgrid(a, b)
    
    #getting the gradients
    u = flow_image[:,:,0]
    v = flow_image[:,:,1]
    
    
    # now lets turn it into grey scale
    plt.figure()
    plt.imshow(image, cmap='gray')
    
    #apply the confidence threshold
    good_idx = np.flatnonzero(confidence>threshmin)
    permuted_indices = np.random.RandomState(seed=10).permutation(good_idx)
    valid_idx = permuted_indices[:3000]
    x = x.ravel()[valid_idx]
    y = y.ravel()[valid_idx]
    u = u.ravel()[valid_idx]
    v = v.ravel()[valid_idx]
    
    # now lets plot the flow field on top of the image
    plt.quiver (x, y, u, v, color='blue', scale=10, width =0.005, angles='xy')
    plt.savefig(f"optical_flow_lines_{threshmin}.png")
    plt.show()
    
    ### STUDENT CODE END ###

    # this function has no return value
    return





    

