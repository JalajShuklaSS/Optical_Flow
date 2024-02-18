import numpy as np

def flow_lk_patch(Ix, Iy, It, x, y, size=5):
    """
    Find the Lucas-Kanade optical flow on a single square patch.
    The patch is centered at (y, x), therefore it generally extends
    from x-size//2 to x+size//2 (inclusive), same for y, EXCEPT when
    exceeding image boundaries!
    
    WARNING: Pay attention to how you index the images! The first coordinate
    is actually the y-coordinate, and the second coordinate is the x-coordinate.
    
    Inputs:
        - Ix: image gradient along the x-dimension - shape: (H, W)
        - Iy: image gradient along the y-dimension - shape: (H, W)
        - It: image time-derivative - shape: (H, W)
        - x: SECOND coordinate of patch center - integer in range [0, W-1]
        - y: FIRST coordinate of patch center - integer in range [0, H-1]
        - size: optional parameter to change the side length of the patch in pixels
    
    Outputs:
        - flow: flow estimate for this patch - shape: (2,)
        - conf: confidence of the flow estimates - scalar
    """

    ### STUDENT CODE START ###
    # 1. Extract the local patch from the images
    local_P_X_a = max(0, x-size//2)
    local_P_X_b = min(Ix.shape[1], x+size//2+1)
    local_P_Y_a = max(0, y-size//2)
    local_P_Y_b = min(Ix.shape[0], y+size//2+1)
    # print(local_P_X_a, local_P_X_b, local_P_Y_a, local_P_Y_b)
    
    local_p_Ix = Ix[local_P_Y_a:local_P_Y_b, local_P_X_a:local_P_X_b]  # shape: (size, size)
    local_p_Iy = Iy[local_P_Y_a:local_P_Y_b, local_P_X_a:local_P_X_b]
    local_p_It = It[local_P_Y_a:local_P_Y_b, local_P_X_a:local_P_X_b]    

    
    # Matrix_A = np.zeros((size**2, 2))
    # Matrix_b = np.zeros((size**2, 1))
    # 2. Construct the matrix A and vector b
    # reordering matrix A components by using flatten()
    p_Ix = local_p_Ix.flatten() #reshaping it
    p_Iy = local_p_Iy.flatten() #reshaping it
    p_It = local_p_It.flatten() #reshaping it
    
    M_A = np.vstack((p_Ix, p_Iy))
    Matrix_A = np.transpose(M_A)
    Matrix_b = -p_It
    
    #solving the linear equation
    ans,_,_,S = np.linalg.lstsq(Matrix_A, Matrix_b, rcond=None)
    
    if ans.size !=2: #if the matrix is empty
        ans = np.array([0,0], dtype=float)
    flow = ans
    
    conf = np.min(S)
 
    ### STUDENT CODE END ###

    return flow, conf


def flow_lk(Ix, Iy, It, size=5):
    """
    Compute the Lucas-Kanade flow for all patches of an image.
    To do this, iteratively call flow_lk_patch for all possible patches.
    
    WARNING: Pay attention to how you index the images! The first coordinate
    is actually the y-coordinate, and the second coordinate is the x-coordinate.
    
    Inputs:
        - Ix: image gradient along the x-dimension - shape: (H, W)
        - Iy: image gradient along the y-dimension - shape: (H, W)
        - It: image time-derivative
    Outputs:
        - image_flow: flow estimate for each patch - shape: (H, W, 2)
        - confidence: confidence of the flow estimates - shape: (H, W)
    """

    ### STUDENT CODE START ###
    # double for-loop to iterate over all patches
    
    H, W = Ix.shape
    image_flow = np.zeros((H, W, 2))
    confidence = np.zeros((H, W))
    
 
    for y in range(H): # iterate over all images
        for x in range(W): 
            flow, con = flow_lk_patch(Ix, Iy, It, x, y, size)
            image_flow[y, x] = flow 
            confidence[y, x] = con
    
    ### STUDENT CODE END ###
    
    return image_flow, confidence
