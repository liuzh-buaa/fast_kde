# Deriche's approximation of Gaussian smoothing
# Adapted from Getreuer's C implementation (BSD license)
# https://www.ipol.im/pub/art/2013/87/gaussian_20131215.tgz
# http://dev.ipol.im/~getreuer/code/doc/gaussian_20131215_doc/gaussian__conv__deriche_8c.html
import math

import numpy as np


def deriche_config(sigma, negative=False):
    # compute causal filter coefficients
    a = np.zeros(shape=5)
    bc = np.zeros(shape=4)
    deriche_causal_coeff(a, bc, sigma)

    # numerator coefficients of the anticausal filter
    ba = np.array([
        0,
        bc[1] - a[1] * bc[0],
        bc[2] - a[2] * bc[0],
        bc[3] - a[3] * bc[0],
        -a[4] * bc[0]
    ])

    # impulse response sums
    accum_denom = 1.0 + a[1] + a[2] + a[3] + a[4]
    sum_causal = (bc[0] + bc[1] + bc[2] + bc[3]) / accum_denom
    sum_anticausal = (ba[1] + ba[2] + ba[3] + ba[4]) / accum_denom

    # coefficients object
    return {
        'sigma': sigma,
        'negative': negative,
        'a': a,
        'b_causal': bc,
        'b_anticausal': ba,
        'sum_causal': sum_causal,
        'sum_anticausal': sum_anticausal
    }


def deriche_causal_coeff(a_out, b_out, sigma):
    K = 4

    alpha = [0.84, 1.8675, 0.84, -1.8675, -0.34015, -0.1299, -0.34015, 0.1299]

    x1 = math.exp(-1.783 / sigma)
    x2 = math.exp(-1.723 / sigma)
    y1 = 0.6318 / sigma
    y2 = 1.997 / sigma
    beta = [-x1 * math.cos(y1), x1 * math.sin(y1), -x1 * math.cos(-y1), x1 * math.sin(-y1),
            -x2 * math.cos(y2), x2 * math.sin(y2), -x2 * math.cos(-y2), x2 * math.sin(-y2)]

    denom = sigma * 2.5066282746310007

    # initialize b/a = alpha[0] / (1 + beta[0] z^-1)
    b = [alpha[0], alpha[1], 0, 0, 0, 0, 0, 0]
    a = [1, 0, beta[0], beta[1], 0, 0, 0, 0, 0, 0]

    for k in range(2, 8, 2):
        b[k] = beta[k] * b[k - 2] - beta[k + 1] * b[k - 1]
        b[k + 1] = beta[k] * b[k - 1] + beta[k + 1] * b[k - 2]
        for j in range(k - 2, 0, -2):
            b[j] += beta[k] * b[j - 2] - beta[k + 1] * b[j - 1]
            b[j + 1] += beta[k] * b[j - 1] + beta[k + 1] * b[j - 2]
        for j in range(0, k + 1, 2):
            b[j] += alpha[k] * a[j] - alpha[k + 1] * a[j + 1]
            b[j + 1] += alpha[k] * a[j + 1] + alpha[k + 1] * a[j]

        a[k + 2] = beta[k] * a[k] - beta[k + 1] * a[k + 1]
        a[k + 3] = beta[k] * a[k + 1] + beta[k + 1] * a[k]
        for j in range(k, 0, -2):
            a[j] += beta[k] * a[j - 2] - beta[k + 1] * a[j - 1]
            a[j + 1] += beta[k] * a[j - 1] + beta[k + 1] * a[j - 2]

    for k in range(K):
        j = k << 1
        b_out[k] = b[j] / denom
        a_out[k + 1] = a[j + 2]


def deriche_conv1d(
        c, src, N,
        stride=1,
        y_causal=None,
        y_anticausal=None,
        h=None,
        d=None,
        init=None
):
    if y_causal is None:
        y_causal = np.zeros(shape=N)
    if y_anticausal is None:
        y_anticausal = np.zeros(shape=N)
    if h is None:
        h = np.zeros(shape=5)
    if d is None:
        d = y_causal
    if init is None:
        init = deriche_init_zero_pad

    stride_2 = stride * 2
    stride_3 = stride * 3
    stride_4 = stride * 4
    stride_N = stride * N

    # initialize causal filter on the left boundary
    init(
        y_causal, src, N, stride,
        c['b_causal'], 3, c['a'], 4, c['sum_causal'], h, c['sigma']
    )

    # filter the interior samples using a 4th order filter. Implements:
    # for n = K, ..., N - 1,
    #   y^+(n) = \sum_{k=0}^{K-1} b^+_k src(n - k)
    #          - \sum_{k=1}^K a_k y^+(n - k)
    # variable i tracks the offset to the nth sample of src, it is
    # updated together with n such that i = stride * n.
    i = stride_4
    for n in range(4, N):
        y_causal[n] = c['b_causal'][0] * src[i] \
                      + c['b_causal'][1] * src[i - stride] \
                      + c['b_causal'][2] * src[i - stride_2] \
                      + c['b_causal'][3] * src[i - stride_3] \
                      - c['a'][1] * y_causal[n - 1] \
                      - c['a'][2] * y_causal[n - 2] \
                      - c['a'][3] * y_causal[n - 3] \
                      - c['a'][4] * y_causal[n - 4]
        i += stride

    # initialize the anticausal filter on the right boundary
    init(
        y_anticausal, src, N, -stride,
        c['b_anticausal'], 4, c['a'], 4, c['sum_anticausal'], h, c['sigma']
    )

    # similar to the causal filter above, the following implements:
    # for n = K, ..., N - 1,
    #   y^-(n) = \sum_{k=1}^K b^-_k src(N - n - 1 - k)
    #          - \sum_{k=1}^K a_k y^-(n - k)
    # variable i is updated such that i = stride * (N - n - 1).
    i = stride_N - stride * 5
    for n in range(4, N):
        y_anticausal[n] = c['b_anticausal'][1] * src[i + stride] \
                          + c['b_anticausal'][2] * src[i + stride_2] \
                          + c['b_anticausal'][3] * src[i + stride_3] \
                          + c['b_anticausal'][4] * src[i + stride_4] \
                          - c['a'][1] * y_anticausal[n - 1] \
                          - c['a'][2] * y_anticausal[n - 2] \
                          - c['a'][3] * y_anticausal[n - 3] \
                          - c['a'][4] * y_anticausal[n - 4]
        i -= stride

    # sum the causal and anticausal responses to obtain the final result
    if c['negative']:
        # do not threshold if the input grid includes negatively weighted values
        i = 0
        for n in range(N):
            d[i] = y_causal[n] + y_anticausal[N - n - 1]
            i += stride
    else:
        # threshold to prevent small negative values due to floating point error
        i = 0
        for n in range(N):
            d[i] = max(0, y_causal[n] + y_anticausal[N - n - 1])
            i += stride

    return d


def deriche_init_zero_pad(
        dest, src, N, stride, b, p, a, q,
        sum_val, h, sigma, tol=0.5
):
    stride_N = abs(stride) * N
    off = stride_N + stride if stride < 0 else 0

    # compute the first q taps of the impulse response, h_0, ..., h_{q-1}
    for n in range(q):
        h[n] = b[n] if n <= p else 0
        for m in range(1, min(q, n) + 1):
            h[n] -= a[m] * h[n - m]

    # compute dest_m = sum_{n=1}^m h_{m-n} src_n, m = 0, ..., q-1
    # note: q == 4
    for m in range(q):
        dest[m] = 0
        for n in range(1, m + 1):
            i = off + stride * n
            if 0 <= i < stride_N:
                dest[m] += h[m - n] * src[i]

    # dest_m = dest_m + h_{n+m} src_{-n}
    cur = src[off]
    max_iter = math.ceil(sigma * 10)
    for n in range(max_iter):
        for m in range(q):
            dest[m] += h[m] * cur

        sum_val -= abs(h[0])
        if sum_val <= tol:
            break

        # Compute the next impulse response tap, h_{n+q}
        h[q] = b[n + q] if n + q <= p else 0
        for m in range(1, q + 1):
            h[q] -= a[m] * h[q - m]

        # Shift the h array for the next iteration
        for m in range(q):
            h[m] = h[m + 1]

    return
