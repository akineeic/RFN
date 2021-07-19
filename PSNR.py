import argparse
import numpy as np
import cv2 as cv
import os
import math
from skimage.measure import compare_psnr as psnr
import skimage.color as sc

def quantize(img):
    return img.clip(0, 255).round().astype(np.uint8)

parser = argparse.ArgumentParser()
parser.add_argument("-sr", "--sr_dir", type=str, default=None)
parser.add_argument("-hr", "--hr_dir", type=str, default=None)


args = parser.parse_args()

PSNR = []
for filename in os.listdir(args.hr_dir):
    print(args.sr_dir + filename.split(".")[0]+"_p.png")
    hr = cv.imread(args.hr_dir + filename)[:, :, [2, 1, 0]]
    sr = cv.imread(args.sr_dir + filename)[:, :, [2, 1, 0]] # filename.split(".")[0]+"_c.png"

    hr = quantize(sc.rgb2ycbcr(hr)[:, :, 0])
    sr = quantize(sc.rgb2ycbcr(sr)[:, :, 0])

    hr = hr[4:-4, 4:-4, ...]
    sr = sr[4:-4, 4:-4, ...]

    tmp = psnr(hr, sr, data_range=255)
    PSNR.append(tmp)

print("psnr is ", PSNR)
print("mean psnr is ", np.mean(np.array(PSNR)))