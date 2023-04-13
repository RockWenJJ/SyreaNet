import json
import os
import argparse
import cv2

import numpy as np
from PIL import Image
from skimage.restoration import denoise_tv_chambolle, estimate_sigma
from datetime import datetime
from tqdm import tqdm
import random

def scale(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def estimate_backscattering(depths, B_inf, beta_B, J_prime, beta_D_prime):
    val = (B_inf * (1 - np.exp(-1 * beta_B * depths))) + (J_prime * np.exp(-1 * beta_D_prime * depths))
    return val

def calculate_beta_D(depths, a, b, c, d):
    return (a * np.exp(b * depths)) + (c * np.exp(d * depths))

def degrade_image(img, depth, B, beta_D, wbalance):
    img = scale(img / np.expand_dims(wbalance, (0, 1)))
    t = np.exp(-beta_D * np.expand_dims(depth, axis=2))
    degrade = img * t + B
    degrade = np.maximum(0.0, np.minimum(1.0, degrade))
    return degrade

def pipeline(img, depth, coefs):
    # get coefficients
    Bcoefs_r, Bcoefs_g, Bcoefs_b = np.array(coefs["Bcoefs_r"]), np.array(coefs["Bcoefs_g"]), np.array(coefs["Bcoefs_b"])
    Dcoefs_r, Dcoefs_g, Dcoefs_b = np.array(coefs["Dcoefs_r"]), np.array(coefs["Dcoefs_g"]), np.array(coefs["Dcoefs_b"])
    wbalance = np.array(coefs['wbalance'])
    # estimate backscattering
    Br = estimate_backscattering(depth, *Bcoefs_r)
    Bg = estimate_backscattering(depth, *Bcoefs_g)
    Bb = estimate_backscattering(depth, *Bcoefs_b)
    B = np.stack([Br, Bg, Bb], axis=2)
    # estimate direct transmission
    beta_D_r = calculate_beta_D(depth, *Dcoefs_r) * 0.5
    beta_D_g = calculate_beta_D(depth, *Dcoefs_g) * 0.5
    beta_D_b = calculate_beta_D(depth, *Dcoefs_b) * 0.5
    beta_D = np.stack([beta_D_r, beta_D_g, beta_D_b], axis=2)
    # degrade images
    degraded = degrade_image(img, depth, B, beta_D, wbalance)
    sigma_est = estimate_sigma(degraded, multichannel=True, average_sigmas=True) / 10.0
    degraded = denoise_tv_chambolle(degraded, sigma_est, multichannel=True)
    return degraded

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', required=True, help='Input in-air images directory')
    parser.add_argument('--depth-dir', required=True, help='Input depth maps of corresponding in-air images')
    parser.add_argument('--out-dir', default=None, help='Ouput directory of synthesized images')
    args = parser.parse_args()
    
    img_files = os.listdir(args.image_dir)
    
    coefs = json.load(open('coeffs.json', 'r'))
    
    if args.out_dir is None:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.out_dir = current_time
    os.makedirs(args.out_dir, exist_ok=True)
    
    for i, img_f in tqdm(enumerate(img_files)):
        # load image
        img = cv2.cvtColor(cv2.imread(os.path.join(args.image_dir, img_f)), cv2.COLOR_BGR2RGB)
        # load depth
        dep_f = os.path.join(args.depth_dir, img_f)
        depth = cv2.imread(os.path.join(args.depth_dir, dep_f), cv2.IMREAD_GRAYSCALE)
        depth = cv2.GaussianBlur(depth, (7, 7), 5) * 1.0
        depth = ((depth - np.min(depth)) / (np.max(depth) - np.min(depth))).astype(np.float32)
        depth = 10. * depth + 2.
        # random select coefficient
        k = random.sample(coefs.keys(), 1)[0]
        degraded = pipeline(img, depth, coefs[k])
        # save images
        im = Image.fromarray((np.clip(np.round(degraded * 255.0), 0, 255)).astype(np.uint8))
        im.save(os.path.join(args.out_dir, img_f), format='png')
