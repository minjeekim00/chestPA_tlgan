import os
import sys
import pickle
import glob
import numpy as np
import tensorflow as tf
import PIL.Image
sys.path.append('./src/model/pggan/')
sys.path.append('./src/model/chestPA_classifier/')
sys.path.append('./src/tl_gan')
import get_normal_images as get_normal_images

import imageio
from src.model.chestPA_classifier.chestPA_cam import plot_cam, classes
from tqdm import tqdm


# path to model code and weight
path_model = './asset_model/x-ray_integrated_20190218_network-snapshot-014000.pkl'
path_gan_explore = './asset_results/axis_explore/'
if not os.path.exists(path_gan_explore):
    os.mkdir(path_gan_explore)
""" get feature direction vector """
path_feature_direction = './asset_results/feature_direction_6/'
pathfile_feature_directions = glob.glob(os.path.join(path_feature_direction, 'feature_direction_*.pkl'))
cases = [4525] #[7386, 4161] #[1987, 3774, 5332]

def create_dst_path():
    path_gan_version = path_gan_explore+method+'_'+version
    if is_weighted:
        path_gan_version+='_weighted'
    if is_orthogonal:
        path_gan_version+='_orthogonal'
    if not os.path.exists(path_gan_version):
        os.mkdir(path_gan_version)
    return path_gan_version

""" play with the latent space """
sess = tf.InteractiveSession()
with open(path_model, 'rb') as file:
    G, D, Gs = pickle.load(file)

for pathfile_feature_direction in pathfile_feature_directions:
    with open(pathfile_feature_direction, 'rb') as f:
        feature_direction_name = pickle.load(f)

    feature_direction = feature_direction_name['direction']
    feature_name = feature_direction_name['name']
    num_feature = feature_direction.shape[1]

    path_basename = os.path.basename(pathfile_feature_direction)
    method = path_basename.split('_')[2]
    version = path_basename.split('_')[3]
    is_weighted = True if 'weighted' in path_basename else False
    is_orthogonal = True if 'orthogonal' in path_basename else False

    if 'linear' in method:
        step_sizes = [0.4, 0.4, 0.4, 0.4, 0.4] 
    elif 'lasso' in method:
        step_sizes = [0.4, 0.4, 0.4, 0.4, 0.8] 
    batch_size = 40

    latents_n = get_normal_images.get_all_images()[cases]# hand-picked image
    
    for cidx, latent in enumerate(latents_n):
        path_gan_version = create_dst_path()
        path_case=path_gan_version+'/'+str(cases[cidx])
        if not os.path.exists(path_case):
            os.mkdir(path_case)
        else: continue
        print(path_case)          
        for cls in range(feature_direction.shape[1]):
            latents = np.random.randn(batch_size, *Gs.input_shapes[0][1:])
            for i, alpha in enumerate(range(batch_size)):
                step_size = step_sizes[(cls)]
                latents[i, :] = latent+(feature_direction[:, cls][None, :][0] * (step_size * alpha))

            # Generate dummy labels (not used by the official networks).
            labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])
            # Run the generator to produce a set of images.
            images = Gs.run(latents, labels)
            images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
            images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC
            images = images.reshape(latents.shape[0],1024,1024)

            # Save images as PNG.
            for idx in range(images.shape[0]):
                dst = os.path.join(path_case, 'img_{}_{}_{:03d}_{}.png'.format(cases[cidx], cls, idx, os.path.basename(path_gan_version)))
                if os.path.exists(dst):
                    continue
                PIL.Image.fromarray(images[idx], 'L').save(dst)

    ##

    img_all = [sorted(glob.glob(os.path.join(path_case, 'img_{}_*.png'.format(i)))) for i in cases]
    img_all = np.array(img_all).reshape(len(cases), len(step_sizes), batch_size).tolist()
    path_cam = "./results_cam/"
    cam_version = os.path.basename(path_gan_version)
    path_dst = ["./results_cam/{}/{}".format(cases[idx], cam_version) for idx in range(len(img_all))]

    for idx, imgs in enumerate(img_all): # case별
        if not os.path.exists(path_cam+str(cases[idx])):
            os.mkdir(path_cam+str(cases[idx]))
        for cidx, cimgs in enumerate(imgs): # class별
            plot_cam(cimgs,plot_fig=True, show_cam=True, save_fig=True, 
                     desired_dir=path_dst[idx])
            cams = sorted(glob.glob(path_dst[0]+'/img_{}_{}_*.png'.format(cases[idx], cidx)))
            images = []
            for filename in cams:
                images.append(imageio.imread(filename))
            imageio.mimsave(path_dst[0]+'/cam_{}_{}.gif'.format(classes[cidx+1], cidx), images)
            

sess.close()