import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
from tqdm import tqdm

# path to model code and weight
path_pg_gan_code = './src/model/pggan'
path_model = './asset_model/x-ray_integrated_20190218_network-snapshot-014000.pkl'
sys.path.append(path_pg_gan_code)

# path to model generated results
path_gen_sample = './asset_results/pggan_x_ray_integrated_norm_sample_pkl/'
path_fake_images = './data/x-ray_integrated_norm/'
if not os.path.exists(path_gen_sample):
    os.mkdir(path_gen_sample)
if not os.path.exists(path_fake_images):
    os.mkdir(path_fake_images)

""" gen samples and save as pickle """
n_batch = 1000
batch_size = 30

with tf.Session() as sess:
    try:
        with open(path_model, 'rb') as file:
            G, D, Gs = pickle.load(file)
    except FileNotFoundError:
        print('before running the code, download pre-trained model to project_root/asset_model/')
        raise

    # Generate latent vectors.
    # latents = np.random.RandomState(1000).randn(1000, *Gs.input_shapes[0][1:]) # 1000 random latents
    # latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]] # hand-picked top-10
    latents = np.random.RandomState(1000).randn(n_batch * batch_size, *Gs.input_shapes[0][1:])
    
    for i_batch in tqdm(range(n_batch)):
        i_sample = i_batch * batch_size

        p_latents = latents[i_batch:i_batch+batch_size]
        labels = np.zeros([p_latents.shape[0]] + Gs.input_shapes[1][1:])

        # Run the generator to produce a set of images.
        images = Gs.run(p_latents, labels)
        images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
        images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC
        images = images.reshape(p_latents.shape[0],1024,1024)
            
            
        # Save images as PNG.
        for idx in range(images.shape[0]):
            path = path_fake_images+'img%06d.png'%int(idx+i_batch)
            if os.path.exists(path):
                continue
            PIL.Image.fromarray(images[idx], 'L').save(path)

        path_pickle = os.path.join(path_gen_sample, 'pggan_x_ray_integrated_norm_{:0>6d}.pkl'.format(i_sample))
        with open(path_pickle, 'wb') as f:
            pickle.dump({'z': p_latents, 'x': images}, f)
            
        del p_latents, labels, images
