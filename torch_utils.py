import torch


def cast_to_torch(item, B: int, device, dtype):
    # Handling torch tensors
    if isinstance(item, torch.Tensor):
        t = item.to(device=device, dtype=dtype) # Cast to right device
        # Add a batch dimension B if if dim is 0
        if t.dim() == 0:
            t = t.expand(B)
        return t
    
    # scalar -> filled (B,) tensor
    return torch.full((B,), float(item), device=device, dtype=dtype)
