import torch
# https://blbadger.github.io/henon-map.html

def henon(X):

    a=1.4
    b=0.3
    x, y = X

    res = torch.stack([1 - a * x ** 2 + y, b * x])
    return res
