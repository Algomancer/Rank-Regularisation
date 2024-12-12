# Rank Regularisation

Rank collapse is a common failure mode in fast teacher self distilation, such as pooled joint embedding predictive archectures with no ema teacher. This simple method can be useful.

```python
def rankreg(x, max_possible_rank, eps=1e-7):
    """
    Encourages maximal rank.    
    Args:
        x: Representations, shape: (batch, embed) or (batch, seq_len, embed)
        max_possible_rank: For normalisation
        eps: Small constant for numerical stability
    
    Returns:
        Loss tensor that when minimized increases effective rank.
        Loss is normalized by maximum possible rank.
    """
    x = x.float()
    
    # Handle both 2D and 3D inputs
    if x.dim() == 3:
        batch_size, seq_len, embed_dim = x.shape
        # Average over sequence dimension
        x = x.mean(dim=1)  # [batch, embed]
    else:
        batch_size, embed_dim = x.shape
    
    # Compute rank
    s = torch.linalg.svdvals(x)
    s_norm = s.norm(1)
    p = s / s_norm
    log_p = torch.log(p + eps)
    entropy = torch.exp(-(p * log_p).sum())
    
    # Normalize by maximum possible rank and negate for loss
    loss = -entropy / max_possible_rank
    
    return loss
```
