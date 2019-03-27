import os
import numpy as np
import h5py

path_gan_sample_img = './asset_results/pggan_x_ray_integrated_norm_sample_jpg/'
path_feature_direction = './asset_results/pg_gan_x_ray_integrated_norm_feature_direction_5/'

# for normal images
filename_normal_z = 'normal_z.h5'
pathfile_normal_z = os.path.join(path_gan_sample_img, filename_normal_z)
with h5py.File(pathfile_normal_z, 'r') as f:
    z_normal = f['z'][:]

def get_single_image():
    """
        num: number of choices
    """
    rand_num = np.random.choice(range(0, len(z_normal)), 1)[0]
    print(rand_num)
    return z_normal[rand_num]


def get_all_images():
    return z_normal