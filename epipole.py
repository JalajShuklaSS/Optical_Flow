import numpy as np

def epipole(flow_x, flow_y, smin, thresh, num_iterations=None):
    """
    Compute the epipole from the flows,
    
    Inputs:
        - flow_x: optical flow on the x-direction - shape: (H, W)
        - flow_y: optical flow on the y-direction - shape: (H, W)
        - smin: confidence of the flow estimates - shape: (H, W)
        - thresh: threshold for confidence - scalar
    	- Ignore num_iterations
    Outputs:
        - ep: epipole - shape: (3,)
    """
    # Logic to compute the points you should use for your estimation
    # We only look at image points above the threshold in our image
    # Due to memory constraints, we cannot use all points on the autograder
    # Hence, we give you valid_idx which are the flattened indices of points
    # to use in the estimation estimation problem 
    good_idx = np.flatnonzero(smin>thresh)
    permuted_indices = np.random.RandomState(seed=10).permutation(
        good_idx
    )
    valid_idx=permuted_indices[:3000]

    ### STUDENT CODE START - PART 1 ###
    
    H, W = flow_x.shape
    w_grid = np.arange(W)-W/2 # the x axis
    h_grid = np.arange(H)-H/2 # the y axis
    x,y = np.meshgrid(w_grid, h_grid, indexing='xy') # the meshgrid
    

    true_x= x.flatten()[valid_idx] # the x coordinates of the valid points
    true_y= y.flatten()[valid_idx] # the y coordinates of the valid points    
   
    c = np.zeros(len(valid_idx))
    X_pix = np.vstack((true_x, true_y, np.ones(len(valid_idx))))
    final_X_pix = np.transpose(X_pix) 
    u_vel = flow_x.flatten()[valid_idx]
    v_vel = flow_y.flatten()[valid_idx]
    u_vect = np.vstack((u_vel, v_vel,c))
    final_u_vect = np.transpose(u_vect)
    A = np.cross(final_X_pix, final_u_vect) 
    u, s, vt = np.linalg.svd(A) # singular value decomposition
    ep = vt[-1] # the last row of vt
    
    

    
    
 
    return ep