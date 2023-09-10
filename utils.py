import numpy as np

def path_pruning(Xh, Xl):
    # remove samples with low variance, i.e., blank spaces and so on
    pvars = np.var(Xh, axis=0) # compute variane in all the images
    threshold = np.percentile(pvars, 10) # compute 10-th percentile 
    idx = pvars > threshold # binary indexing
    Xh = Xh[idx, :] # may need to change this
    Xl = Xl[idx, :]
    return Xh, Xl



