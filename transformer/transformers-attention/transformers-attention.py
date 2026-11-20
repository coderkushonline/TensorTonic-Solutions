import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    attention = torch.softmax(Q @ K.transpose(-2, -1)/( K.size(-1) ** 0.5), dim=-1) @ V

    return attention