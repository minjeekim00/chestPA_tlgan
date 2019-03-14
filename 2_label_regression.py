""" functions to regress y (labels) based on z (latent space) """
import os
import glob
import numpy as np
import pickle
import h5py
import pandas as pd
import sys
import tensorflow as tf
import PIL.Image
import datetime
import glob

import src.misc as misc
import src.tl_gan.feature_axis as feature_axis
import matplotlib.pyplot as plt
%matplotlib inline


""" get y and z from pre-generated files """
path_gan_sample_img = './asset_results/pggan_x_ray_integrated_norm_sample_jpg/'
#path_celeba_att = './data/raw/celebA_annotation/list_attr_celeba.txt'
path_feature_direction = './asset_results/pg_gan_x_ray_integrated_norm_feature_direction_5/'

filename_sample_y = 'sample_y.h5'
filename_sample_z = 'sample_z.h5'

pathfile_y = os.path.join(path_gan_sample_img, filename_sample_y)
pathfile_z = os.path.join(path_gan_sample_img, filename_sample_z)

with h5py.File(pathfile_y, 'r') as f:
    y = f['y'][:]
with h5py.File(pathfile_z, 'r') as f:
    z = f['z'][:]
    
    
# for normal images
filename_normal_z = 'normal_z.h5'
pathfile_normal_z = os.path.join(path_gan_sample_img, filename_normal_z)
with h5py.File(pathfile_normal_z, 'r') as f:
    z_normal = f['z'][:]
    
    
def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
#y_onehot = np.eye(len(y), 5, dtype=np.int8)[y.reshape(-1)]
#y_name = ['00Normal', '01Nodule', '03Consolidation', '04InterstitialOpacity','10PleuralEffusion']
y_onehot = np.eye(len(y), 4, dtype=np.int8)[(y - 1).reshape(-1)]
y_name = ['01Nodule', '03Consolidation', '04InterstitialOpacity','10PleuralEffusion']

##
""" regression: use latent space z to predict features y """
method ='linear'
feature_slope = feature_axis.find_feature_axis(z, y_onehot, method=method)

""" normalize the feature vectors """
yn_normalize_feature_direction = True
if yn_normalize_feature_direction:
    feature_direction = feature_axis.normalize_feature_axis(feature_slope)
else:
    feature_direction = feature_slope
    
""" save_regression result to hard disk """
if not os.path.exists(path_feature_direction):
    os.mkdir(path_feature_direction)
    
pathfile_feature_direction = os.path.join(path_feature_direction, 'feature_direction_{}_{}.pkl'.format(method, misc.gen_time_str()))
dict_to_save = {'direction': feature_direction, 'name': y_name}
with open(pathfile_feature_direction, 'wb') as f:
    pickle.dump(dict_to_save, f)

""" disentangle correlated feature axis """
pathfile_feature_direction = glob.glob(os.path.join(path_feature_direction, 'feature_direction_*.pkl'))[-1]

with open(pathfile_feature_direction, 'rb') as f:
    feature_direction_name = pickle.load(f)

feature_direction = feature_direction_name['direction']
feature_name = np.array(feature_direction_name['name'])

len_z, len_y = feature_direction.shape

feature_direction_disentangled = feature_axis.disentangle_feature_axis_by_idx(
    feature_direction, idx_base=range(len_y), idx_target=None)

#feature_axis.plot_feature_cos_sim(feature_direction_disentangled, feature_name=feature_name)