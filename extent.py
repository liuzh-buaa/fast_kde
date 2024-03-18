def extent(data, pad=0):
    # lo = None
    # hi = None
    # for i in range(len(data)):
    #     v = x(data[i], i, data)
    #     if v is not None:
    #         if lo is None:
    #             if v >= v:
    #                 lo = hi = v
    #         else:
    #             if v < lo:
    #                 lo = v
    #             if v > hi:
    #                 hi = v
    # return [lo - pad, hi + pad]
    return [data.min() - pad, data.max() + pad]
