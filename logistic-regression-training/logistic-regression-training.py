import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 500) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    y=np.array(y)
    w = np.zeros(X.shape[1])
    b = 0
    for _ in range(steps):
      N = len(y)
    
      # Forward pass
      z = X @ w.T + b
      y_pred = _sigmoid(z)
      
      # Backward pass (gradients)
      error = y_pred - y          # Shape: (N,)
      dw = (1 / N) * (np.array(X).T @ error)     # Shape: (features,)
      db = (1 / N) * np.sum(error)     # Scalar
      
      w -= lr * dw
      b -= lr * db
    return (w, b)

    
# train_logistic_regression([[0],[1],[2],[3]], [0,0,1,1])

    