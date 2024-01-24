import torch
# csc.ucdavis.edu/~chaos/courses/nlp/Software/PartG_Code/BakersMap.py


def baker(X):

    a=0.3
    x, y = X

    # Assume (x,y) is on [0,1] x [0,1]
    y = a* y
    if x > 0.5:
        y = y + 0.5
    
    x = 2.0 * x
    while x > 1.0:
        x = x- 1.0 

    
    # res = torch.stack([1 - a * x ** 2 + y, b * x])
    return torch.stack([x, y])