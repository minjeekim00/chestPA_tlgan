""" module to generate iamges from pg-gan """

import os
import sys
import numpy as np
import tensorflow as tf
import PIL

path_pg_gan_code = './src/model/pggan'
path_model = './asset_model/x-ray_integrated_20190218_network-snapshot-014000.pkl'
sys.path.append(path_pg_gan_code)

len_z = 512
len_dummy = 0


def gen_single_img(z=None, Gs=None):
    """
    function to generate image from noise
    :param z:  1D array, latent vector for generating images
    :param Gs: generator network of GAN
    :return:   one gray image, H*W
    """

    if z is None:  # if input not given
        z = np.random.randn(len_z)
    if len(z.shape) == 1:
        z = z[None, :]
    dummy = np.zeros([z.shape[0], len_dummy])
    images = Gs.run(z, dummy)
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC
    images = images.reshape(z.shape[0],1024,1024)
    return images[0]


def save_img(img, pathfile):
    PIL.Image.fromarray(img, 'L').save(pathfile)