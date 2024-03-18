import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity

from bin1d import bin1d
from deriche import deriche_config, deriche_conv1d
from extent import extent as density_extent
from nrd import nrd


def density1d(data, weight=None, bandwidth=None, adjust=1, extent=None, pad=3, bins=512):
    if weight is None:
        weight = np.ones(shape=(len(data))) / len(data)

    if bandwidth is None:
        bandwidth = adjust * nrd(data)

    if extent is None:
        lo, hi = density_extent(data, pad * bandwidth)
    else:
        lo, hi = extent

    grid = bin1d(data, weight, lo, hi, bins)
    delta = (hi - lo) / (bins - 1)
    neg = any(v < 0 for v in grid)

    config = deriche_config(bandwidth / delta, neg)
    d = deriche_conv1d(config, grid, bins)
    return d, lo, hi


if __name__ == '__main__':
    # 1d density estimation, with automatic bandwidth and extent
    # resulting estimator d1 is an object and also an iterable
    data = np.random.randn(100)
    d, lo, hi = density1d(data)
    print(d.sum())
    data = data.reshape(-1, 1)
    kde = KernelDensity()
    kde.fit(data)
    x_label = np.linspace(start=lo, stop=hi, num=512)
    x_label = x_label.reshape(-1, 1)
    dens = np.exp(kde.score_samples(x_label))
    print(dens.sum())
    plt.plot(x_label, d)
    plt.title('fast_kde')
    plt.show()
    plt.close()
    plt.plot(x_label, dens)
    plt.title('kde')
    plt.show()
    plt.close()
