import torch
import numpy as np
import random

def set_seed(seed=42):
    '''Sets the seed of different libraries to ensure reproducibility'''
    # Python's built-in random module
    random.seed(seed)
    
    # NumPy's random generator
    np.random.seed(seed)
    
    # PyTorch's random generator
    torch.manual_seed(seed)
    
    # If using CUDA (GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in certain operations (may affect performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1)