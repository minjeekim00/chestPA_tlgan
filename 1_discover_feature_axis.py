import os
import sys
import time
import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
import datetime
import glob
import h5py
import shutil
from tqdm import tqdm

sys.path.append('./src/model/chestPA_classifier')
from model import classifier
from model.classifier import classes
import chestPA_test


# sample curation
def select_samples(dict_zx): # sigmoid 0.9 이상값만 선별 
    z = dict_zx['z']
    x = dict_zx['x']
    list_pred = chestPA_test.test(x)
    list_z = []
    list_y = []
    for i, pred in enumerate(list_pred):
        idx = np.argmax(pred)
        tmp_pred = list(pred[:]) # copy pred
        tmp_pred.pop(idx) # list of predictions except target
        
        if idx == 0: continue # exclude normal class
        if pred[idx] > 0.9 and all(p < 0.1 for p in tmp_pred):
            list_z.append(z[i])
            list_y.append(idx)
    
    if len(list_z) != 0:
        list_z = np.vstack(list_z)
        list_y = np.asarray(list_y, dtype=np.int8).T
        #print("{} out of {} are selected".format(len(list_z), len(z)))
        return list_z, list_y, list_pred
    else:
        #print("nothing was selected")
        return None, None, list_pred
    
    
# path to model generated results# path to model generated results

path_gan_sample = './asset_results/pggan_x_ray_integrated_norm_sample_pkl/'
path_pred_sample = os.path.join(path_gan_sample, "predictions")
if not os.path.exists(path_pred_sample):
    os.mkdir(path_pred_sample)
list_pkl = sorted(glob.glob(path_gan_sample+'*.pkl'))

# initialization
arr_z = np.empty([1, 512], dtype=np.float32)
arr_y = np.empty([1], dtype=np.int8)

for file_pkl in tqdm(list_pkl[:]):
    with open(file_pkl, 'rb') as f:
        dict_zx = pickle.load(f)
        z, y, preds = select_samples(dict_zx)
        if z is not None and y is not None:
            arr_z = np.concatenate((arr_z, z), 0)
            arr_y = np.concatenate((arr_y, y), 0)
        basename_pkl = os.path.splitext(os.path.basename(file_pkl))[0]
        path_preds = os.path.join(path_pred_sample, basename_pkl+'.npy')
        np.save(path_preds, preds) # saving predictions
        
## save as h5file
path_sample_jpg = './asset_results/pggan_x_ray_integrated_norm_sample_jpg/'
h5files = ['sample_z.h5', 'sample_y.h5']
arrays = [arr_z, arr_y]
data = ['z', 'y']
        
for i, h5file in enumerate(h5files):
    h5path = path_sample_jpg+h5file
    with h5py.File(h5path, 'w') as hf:
        hf.create_dataset(data[i], data=arrays[i][1:])
hf.close()