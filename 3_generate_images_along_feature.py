import os
import sys
import pickle
import glob
import numpy as np
import tensorflow as tf
import PIL.Image
import src.tl_gan.get_normal_images as get_normal_images


""" start tf session and load GAN model """

# path to model code and weight
path_pg_gan_code = './src/model/pggan'
path_model = './asset_model/x-ray_integrated_20190218_network-snapshot-014000.pkl'
sys.path.append(path_pg_gan_code)
path_gan_explore = './asset_results/pggan_x_ray_integrated_norm_axis_explore/'

""" get feature direction vector """
path_feature_direction = './asset_results/pg_gan_x_ray_integrated_norm_feature_direction_5/'
pathfile_feature_direction = glob.glob(os.path.join(path_feature_direction, 'feature_direction_*.pkl'))[-1]

with open(pathfile_feature_direction, 'rb') as f:
    feature_direction_name = pickle.load(f)

feature_direction = feature_direction_name['direction']
feature_name = feature_direction_name['name']
num_feature = feature_direction.shape[1]


""" play with the latent space """
sess = tf.InteractiveSession()
with open(path_model, 'rb') as file:
    G, D, Gs = pickle.load(file)
    
    
step_sizes = [1, 0.25 ,0.25, 0.25] 
batch_size = 40

#latents_n = get_normal_images.get_single_image()
handpicked = [1987, 3774, 5332]
latents_n = get_normal_images.get_all_images()[handpicked]# hand-picked image


for i_latent, latent_n in enumerate(latents_n):
    for i_feature in range(feature_direction.shape[1]):
        latents = np.random.randn(batch_size, *Gs.input_shapes[0][1:])
        for i, alpha in enumerate(range(batch_size)):
            step_size = step_sizes[(i_feature)]
            latents[i, :] = latent_n + (feature_direction[:, i_feature][None, :][0] * (step_size * alpha))

        # Generate dummy labels (not used by the official networks).
        labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])
        # Run the generator to produce a set of images.
        images = Gs.run(latents, labels)
        images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
        images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC
        images = images.reshape(latents.shape[0],1024,1024)

        import datetime
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save images as PNG.
        for idx in range(images.shape[0]):
            PIL.Image.fromarray(images[idx], 'L').save(os.path.join(path_gan_explore, 'img_{}_{}_{}.png'.format(handpicked[i_latent], i_feature, idx)))
        #np.save(os.path.join(path_gan_explore, 'img_{}_{}.pkl'.format(time_str, i_feature)), labels)

##
sess.close()




#### for generating CAM images

import imageio
from src.model.chestPA_classifier.cam import plot_cam, classes


path_gan_explore = './asset_results/pggan_x_ray_integrated_norm_axis_explore/'

n = batch_size * len(step_sizes)
b = batch_size
# hand-picked images
total_images = [sorted(glob.glob(os.path.join(path_gan_explore, 'img_{}_*.png'.format(i))), key=os.path.getmtime) for i in handpicked]

## generate CAM
for case_idx, images in enumerate(total_images): # case별
    for class_idx, images_per_class in enumerate([images[0:b], images[b:b*2], images[b*2:b*3], images[b*3:]]): # class별
        plot_cam(dataset_test=images_per_class, plot_fig=True, show_cam=True, save_fig=True, desired_dir="./results/{}".format(handpicked[case_idx]))
        cams_by_handpicked = sorted(glob.glob('./results/{}/img_{}_{}_*.png'.format(handpicked[case_idx], handpicked[case_idx], class_idx)), key=os.path.getmtime)
        images = []
        for filename in cams_by_handpicked:
            images.append(imageio.imread(filename))
        imageio.mimsave('./results/{}/cam_{}.gif'.format(handpicked[case_idx], classes[class_idx+1]), images)