def min_max_scaling(data: list) -> list:
    """
    Returns each data column scaled to the range from 0 through 1.
    """
    # Write code here
    rows = len(data)
    cols = len(data[0])

    for i in range(cols):
        max = -1e10
        min = 1e10
        for j in range(rows):
            num = data[j][i]
            min = num if min > num else min
            max = num if max < num else max

        for j in range(rows):
            data[j][i] = (data[j][i] - min) / (max - min) if max != min else 0
    return data