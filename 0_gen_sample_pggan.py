import os
import sys
import pickle
import random
import time
import numpy as np
import tensorflow as tf
import PIL.Image
from tqdm import tqdm

# path to model code and weight
path_pg_gan_code = './src/model/pggan'
path_model = './asset_model/x-ray_integrated_20190218_network-snapshot-014000.pkl'
sys.path.append(path_pg_gan_code)

# path to model generated results
path_gen_sample = './asset_results/sample_pkl/'
path_fake_images = './data/images/'
if not os.path.exists(path_gen_sample):
    os.mkdir(path_gen_sample)
if not os.path.exists(path_fake_images):
    os.mkdir(path_fake_images)

""" gen samples and save as pickle """
n_batch = 1000
batch_size = 30

def get_randnum(time):
    return int((time - int(time))* 1000000.0)

with tf.Session() as sess:
    try:
        with open(path_model, 'rb') as file:
            G, D, Gs = pickle.load(file)
    except FileNotFoundError:
        print('before running the code, download pre-trained model to project_root/asset_model/')
        raise
        
    latents = []
    images = []
    for idx in tqdm(range(n_batch * batch_size)[:]):
        
        latent = np.random.RandomState(get_randnum(time.time())).randn(1, *Gs.input_shapes[0][1:])
        label = np.zeros([1] + Gs.input_shapes[1][1:])
        image = Gs.run(latent, label)
        image = np.clip(np.rint((image + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)# [-1,1] => [0,255]
        image = image.reshape(1024, 1024)
        path = path_fake_images+'img%06d.png'%int(idx)
        PIL.Image.fromarray(image, 'L').save(path)
        
        latents.append(latent)
        images.append(image)
        
        if idx !=0 and idx % 30 == 0:
            path_pickle = os.path.join(path_gen_sample, 'pggan_x_ray_integrated_norm_{:0>6d}.pkl'.format(idx))
            latents = np.asarray(latents, dtype=np.float32)
            images = np.asarray(images, dtype=np.uint8)
            with open(path_pickle, 'wb') as f:
                pickle.dump({'z': latents, 'x': images}, f)
            # re-initialize
            latents = []
            images = []