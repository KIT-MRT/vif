import argparse
import glob

from tqdm import tqdm
import skimage
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir", type=str)
    args = parser.parse_args()

    images_dir = args.images_dir

    means = []
    stds = []
    for image_file in tqdm(glob.glob(f"{images_dir}*/*/*/*.jpeg")):
        image = skimage.io.imread(image_file)
        mean = image[:,:,0].mean(), image[:,:,1].mean(), image[:,:,2].mean()
        std = image[:,:,0].std(), image[:,:,1].std(), image[:,:,2].std()

        means.append(mean)
        stds.append(std)
    
    print(1/(255/np.array(means).mean(axis=0)))
    print(1/(255/np.array(stds).mean(axis=0)))
