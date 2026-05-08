import numpy as np

def compute_error(pred, gt):
    pred = pred[0,0].cpu().numpy() # goes from [batch, channel, H, W] to [H,W]
    gt   = gt[0].cpu().numpy() # goes from [batch, H, W] to [H,W]

    # 1. Flatten heatmap and find max index ; 2. Convert flat index to (row, col) : row = idx // width and col = idx%width
    # unravel_index maps it back to 2d grid
    r_pred, c_pred = np.unravel_index(pred.argmax(), pred.shape)
    r_gt,   c_gt   = np.unravel_index(gt.argmax(),   gt.shape)

    return np.sqrt(((r_gt - r_pred)*0.1)**2 + ((c_gt - c_pred)*0.1)**2) # factor 0.1 converts pixel differences to meters