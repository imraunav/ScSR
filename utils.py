import numpy as np
import cv2
from scipy.signal import convolve2d

def patch_pruning(Xh, Xl):
    # remove samples with low variance, i.e., blank spaces and so on
    pvars = np.var(Xh, axis=1) # compute variane in all the images
    threshold = np.percentile(pvars, 10) # compute 10-th percentile 
    idx = pvars > threshold # binary indexing
    # print(Xh.shape, idx.shape)
    Xh = Xh[idx, :] # may need to change this based on how the matrices are defined
    Xl = Xl[idx, :]
    return Xh, Xl

def patchify(im, patchsize=3):
    '''
    Return patches of [patchsize] from images 
    '''
    patches = []
    # print("Patchify", im.shape)
    l, b = im.shape[0], im.shape[1]
    for i in range(0, l, patchsize):
        for j in range(0, b, patchsize):
            patch = im[i:i+patchsize, j:j+patchsize]
            patches.append(patch)
    return np.array(patches)

def rescale(im, scale):
    '''
    resize by a factor
    '''
    new_im = cv2.resize(im, (0,0), fx=scale, fy=scale) 
    return new_im

def resize(im, dim):
    '''
    resize by dim
    '''
    new_im = cv2.resize(im, dim)
    return new_im

def read_im(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    im = im / 255.0
    return im.astype(np.float32)

def write_im(im, path):
    im = im * 255.0
    im = im.round()
    im = im.astype(np.uint8)
    cv2.imwrite(path, im)

def bgr2ycrcb(im):
    # print(im.shape)
    return cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)


def lr_features(lr):
    '''
    computing first and second order gradients
    '''
    hf1 = np.array(
        [[-1, 0, 1],
         [-1, 0, 1],
         [-1, 0, 1]]
    )
    vf1 = hf1.T
    hf2 = np.array(
        [[1, 0, -2, 0, 1],
        [1, 0, -2, 0, 1],
        [1, 0, -2, 0, 1]]
    )
    vf2 = hf2.T

    # gradients
    lr_g11 = convolve2d(lr, hf1, 'same')
    lr_g12 = convolve2d(lr, vf1, 'same')

    lr_g21 = convolve2d(lr, hf2, 'same')
    lr_g22 = convolve2d(lr, vf2, 'same')

    lr_f = np.empty((*lr.shape[:2], 4))
    lr_f[:,:,0] = lr_g11
    lr_f[:,:,1] = lr_g12
    lr_f[:,:,2] = lr_g21
    lr_f[:,:,3] = lr_g22
    return lr_f


def process_patch(hr, patchsize, scale):
    '''
    Process patches for a given image
    '''
    lr = rescale(hr, scale)
    lr = resize(lr, hr.shape[:2]) #interpolation
    lr_feat = lr_features(lr)
    patchsize = 3
    hp = patchify(hr, patchsize)
    hp = hp.reshape((hp.shape[0], -1))
    lp = patchify(lr_feat, patchsize)
    lp = lp.reshape((lp.shape[0], -1))
       
    return hp, lp

