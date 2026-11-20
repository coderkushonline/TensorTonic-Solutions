import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    pos_vector = np.arange(seq_length, dtype="float64").reshape(-1, 1)

    two_i = np.arange(0, d_model, 2)
    divide_term = np.power(10000, two_i/d_model)

    #  Initialize the matrix and fill even/odd columns using slicing
    pe = np.zeros((seq_length, d_model))
    pe[:, 0::2] = np.sin(pos_vector / divide_term)  # Even columns (0, 2) -> sine
    pe[:, 1::2] = np.cos(pos_vector / divide_term)  # Odd columns (1, 3)  -> cosine

    return pe
            