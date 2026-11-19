from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    # Write code here
    x = np.array(x, dtype="float64")
    n = len(x)
    mean = np.sum(x) / n
    s = np.sort(x)
    median = (s[n // 2 -1] + s[n // 2]) / 2if n % 2 == 0 else s[n // 2]
    mode = Counter(x).most_common(1)[0][0]
    op = {"mean": float(mean), "median": float(median), "mode": float(mode)}
    return op
    