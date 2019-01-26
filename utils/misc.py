from .imports import *

def zero_boundary(df):
    zeros = (df == 0).astype('int')
    boundary = zeros.diff()

    # handle boundary condition (no pun intended)
    if zeros.iloc[0] == 1: boundary.iloc[0] = 1
    if zeros.iloc[-1] == 1: boundary.iloc[-1] = -1

    begins = np.argwhere(np.array((boundary == 1))).squeeze()
    ends = np.argwhere(np.array((boundary == -1))).squeeze()
    num_zeros = [e - b for b, e in zip(begins, ends)]

    return begins, ends, num_zeros 
