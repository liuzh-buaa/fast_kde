import numpy as np


def nrd(data):
    values = np.array(data)
    values.sort()
    sd = values.std()
    q1 = np.quantile(values, 0.25)
    q3 = np.quantile(values, 0.75)

    n = len(values)
    h = (q3 - q1) / 1.34
    v = min(sd, h) or sd or abs(q1) or 1  # ``a or b'':= a if a != 0 else b

    return 1.06 * v * (n ** -0.2)


# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
def stdev(values):
    n = len(values)
    count, mean, sum_val = 0, 0, 0
    for i in range(n):
        count += 1
        value = values[i]
        delta = value - mean
        mean += delta / count
        sum_val += delta * (value - mean)
    return (sum_val / (count - 1)) ** 0.5 if count > 1 else float('nan')


# Note ``values'' have been sorted by default
def quantile(values, p):
    n = len(values)

    if n == 0:
        return float('nan')
    if p <= 0 or n < 2:
        return values[0]
    if p >= 1:
        return values[n - 1]

    i0 = (n - 1) * p
    i1 = int(i0)
    return values[i1] + (values[i1 + 1] - values[i1]) * (i0 - i1)


if __name__ == '__main__':

    for i in range(10):
        test_data = np.random.randint(1, 101, size=(15,))
        test_data.sort()
        print(f'Epoch {i}: np={test_data.std():.3f}, stdev={stdev(test_data):.3f}; '
              f'np={np.quantile(test_data, 0.25)}/{np.quantile(test_data, 0.75)}, '
              f'quantile={quantile(test_data, 0.25)}/{quantile(test_data, 0.75)}; '
              f'test_data={test_data}')
