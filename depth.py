import numpy as np

def depth(flow, confidence, ep, K, thres=10):
    """
    Compute the depth map from the flow and confidence map.
    
    Inputs:
        - flow: optical flow - shape: (H, W, 2)
        - confidence: confidence of the flow estimates - shape: (H, W)
        - ep: epipole - shape: (3,)
        - K: intrinsic calibration matrix - shape: (3, 3)
        - thres: threshold for confidence (optional) - scalar
    
    Output:
        - depth_map: depth at every pixel - shape: (H, W)
    """
    depth_map = np.zeros_like(confidence)

    ### STUDENT CODE START ###
    
    # 1. Find where flow is valid (confidence > threshold)
    a = confidence > thres
    
    # 2. Convert these pixel locations to normalized projective coordinates
    locations = np.array(np.where(a))
    locations_ones = np.ones(locations.shape[1])
    coordinates = np.vstack((locations, locations_ones))
    coordinates = np.matmul(np.linalg.inv(K), coordinates) 
    
    # 3. Same for epipole and flow vectors
    ep = ep.reshape(3, 1)
    normalized_ep = np.matmul(np.linalg.inv(K), ep)
    normalized_flow = flow[a]
    
    # 4. Now find the depths using the formula from the lecture slides
    delta_points = np.subtract(coordinates, normalized_ep)
    final_depth = np.linalg.norm(delta_points, axis=0) / np.linalg.norm(normalized_flow, axis=1)
    depth_map[a] = final_depth
    
    ### STUDENT CODE END ###
    
    
    ## Truncate the depth map to remove outliers
    
    # require depths to be positive
    truncated_depth_map = np.maximum(depth_map, 0) 
    valid_depths = truncated_depth_map[truncated_depth_map > 0]
    
    # You can change the depth bound for better visualization if your depth is in a different scale
    depth_bound = valid_depths.mean() + 10 * np.std(valid_depths)
    print(f'depth bound: {depth_bound}')

    # set depths above the bound to 0 and normalize to [0, 1]
    truncated_depth_map[truncated_depth_map > depth_bound] = 0
    truncated_depth_map = truncated_depth_map / truncated_depth_map.max()

    return truncated_depth_map
