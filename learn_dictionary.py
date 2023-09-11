from sklearn.decomposition import MiniBatchDictionaryLearning
import cv2
import numpy as np
import pickle
from tqdm import tqdm 
import os
from sklearn.preprocessing import normalize
import time


from utils import read_im, process_patch, patch_pruning, bgr2ycrcb

def mycallback(arg):
    # print(type(arg))
    # print(arg.keys())
    print("Current cost: ", arg['current_cost'])
    # print("Dictionary: ", type(arg['dictionary'])) #np.nparray
    save(arg['dictionary'], './dictionaries/ckpt/ckpt.pkl')

def save(D, name):
    with open(name, mode='wb') as file:
        pickle.dump(D, file)

def main():
    n_atoms = 1024
    batch_size = 256
    max_iter = 1000
    patchsize = 3
    up_scale = 3

    print("Reading samples from /trainset/ ...")
    path = './trainset'
    Xh = []
    Xl = []
    for filename in tqdm(os.listdir(path)):
        filepath = os.path.join(path, filename)
        im = read_im(filepath) 
        ycrcb = bgr2ycrcb(im)
        y = ycrcb[:, :, 0]
        hp, lp = process_patch(y, patchsize=patchsize, scale=3)
        Xh.extend(hp)
        Xl.extend(lp)
    print("Done!")
    Xh = np.array(Xh)
    Xl = np.array(Xl)

    Xh, Xl = patch_pruning(Xh, Xl)

    print('Training dictionary...')
    #preprocessing
    Xh = normalize(Xh, axis=1)
    Xl = normalize(Xl, axis=1)
    N = Xh.shape[1]
    M = Xl.shape[1]
    coupled_patches = np.concatenate((Xh*np.sqrt(N), Xl*np.sqrt(M)), axis=1)
    tic = time.perf_counter()
    dict_learn_compact = MiniBatchDictionaryLearning(n_components=n_atoms,batch_size=batch_size, 
                                            verbose=True,
                                            transform_algorithm='omp', 
                                            max_iter=max_iter,
                                            n_jobs=-1)
                                            # callback=mycallback)
    dict_learn_compact.fit(coupled_patches)
    Dc = dict_learn_compact.components_
    toc = time.perf_counter()
    print(f'Time taken to learn the coupled dictionary = {(toc-tic)/60:.5} mins')

    # decoupling
    Dh = Dc[:, :N]
    Dl = Dc[:, N:]
    save(Dh, f"./dictionaries/Dh_{n_atoms}_{patchsize}_{max_iter}.pkl")
    save(Dl, f"./dictionaries/Dl_{n_atoms}_{patchsize}_{max_iter}.pkl")

    print("Dictionaries saved!")

if __name__ == "__main__":
    main()
