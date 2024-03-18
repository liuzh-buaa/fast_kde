import numpy as np


def bin1d(data, weight, lo, hi, n):
    grid = np.zeros(shape=n)
    delta = (hi - lo) / (n - 1)

    for i in range(len(data)):
        p = (data[i] - lo) / delta
        u = int(p)
        v = u + 1

        if 0 <= u < n and v < n:
            grid[u] += (v - p) * weight[i]
            grid[v] += (p - u) * weight[i]
        elif u == -1:
            grid[v] += (p - u) * weight[i]
        elif v == n:
            grid[u] += (v - p) * weight[i]

    return grid
