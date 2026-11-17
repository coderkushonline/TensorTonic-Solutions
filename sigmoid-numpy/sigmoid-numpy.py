import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    # Write code here

    x = np.array(x)
    sig_fn = np.vectorize(lambda x: 1/(1 + np.power(np.e, -x)))
    opt_value = sig_fn(x)
    
    return opt_value