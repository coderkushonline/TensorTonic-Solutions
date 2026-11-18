import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    # initialize position formula
    sine = lambda angle : np.sin(angle)
    cosine = lambda angle : np.cos(angle)

    #initialize array
    pe = np.zeros((seq_len, d_model))
    i = np.arange(d_model).reshape(1, -1) # for getting value of i in formula
    pos = np.arange(seq_len).reshape(-1, 1) # for getting value of pos in the formula
    
    angles = pos / np.power(base, (2 * (i // 2)) / d_model)
    pe = np.where(i % 2 == 0, sine(angles), cosine(angles))
    return pe