import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse

def data_sample(source, destination, sample_size=96):
    '''
    Samples data from a dataset of images for learning dictionary
    '''
    print("Sampling from large set...\n")
    name = 0 # something to name the image samples by
    for img in tqdm(os.listdir(source)):
        img = os.path.join(source, img)
        # print(img)
        im = cv2.imread(img, cv2.IMREAD_COLOR)
        try:
            if im == None:
                continue
        except:
            pass
        l, b, d = im.shape
        for _ in range(8): # extract 8 patches from single high res image
            x = np.random.randint(0, l-sample_size)
            y = np.random.randint(0, b-sample_size)
            sample = im[x:x+sample_size, y:y+sample_size, :]
            filename = os.path.join(destination, f'{name}.png')
            name += 1
            cv2.imwrite(filename, sample)
    print("Done!")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str, help="Directory of large dataset")
    parser.add_argument("destination", type=str, help="Directory to store samples")
    args = parser.parse_args()

    data_sample(args.source, args.destination)