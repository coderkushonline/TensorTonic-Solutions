import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    N = len(seqs)
    L = max_len if max_len else max([len(seq) for seq in seqs], default=0)
    
    result = np.array([np.pad(r, (0, L - len(r)), constant_values=pad_value) if (L - len(r)) >= 0 else r[:max_len] for r in seqs], dtype=np.int32).reshape(N, L)
    return result


    