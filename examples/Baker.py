import torch
import math
# csc.ucdavis.edu/~chaos/courses/nlp/Software/PartG_Code/BakersMap.py


# def baker(X):

#     a=0.3
#     x, y = X

#     # Assume (x,y) is on [0,1] x [0,1]
#     y = a* y
#     if x > 0.5:
#         y = y + 0.5
    
#     x = 2.0 * x
#     while x > 1.0:
#         x = x- 1.0 

    
#     return torch.stack([x, y])

def baker(X):

    '''From "Efficient Computation of Linear Response of Chaotic Attractors with
One-Dimensional Unstable Manifolds" by Chandramoorthy et al. 2022 '''

    x, y = X #[0., 0., 0.5, 0.]
    s1 = 0
    s2 = 0.5
    s3 = 0.6
    s4 = 0.
    x = 2*x + (s1 + s2*torch.sin((2*y)/2))*torch.sin(x) - torch.floor(x/torch.pi)*2*torch.pi
    y = (y + (s4 + s3*torch.sin(x))*torch.sin(2*y) + torch.floor(x/torch.pi)*2*torch.pi)/2

    x = x % (2*torch.pi)
    y = y % (2*torch.pi)
    
    return torch.stack([x, y])
