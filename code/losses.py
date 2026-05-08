import torch.nn.functional as F

# Baseline loss
def baseline_loss(loc_pred, cons_pred, L, B, lambda_c=0.1):

    # L1 location loss (weighted)
    loss_loc = ((loc_pred - L).abs() * (5*L + 0.1)).mean()

    # L1 consistency loss
    loss_cons = F.l1_loss(cons_pred, B)

    total = loss_loc + lambda_c * loss_cons

    return total, loss_loc.item(), loss_cons.item()


# UNet/ UNet+Transformer loss
def unet_loss(loc_pred, cons_pred, L, B,
              lambda_cons=0.1, lambda_l1=5e-4):
    
    # Weighted MSE : weights pixels near the GT peak
    weight    = 5.0 * L + 0.1
    loss_loc  = (F.mse_loss(loc_pred, L, reduction='none') * weight).mean()

    # L1 sparsity 
    loss_l1   = lambda_l1 * loc_pred.abs().mean()

    # Consistency loss 
    loss_cons = F.mse_loss(cons_pred, B)

    total = loss_loc + loss_l1 + lambda_cons * loss_cons
    return total, loss_loc.item(), loss_cons.item()
