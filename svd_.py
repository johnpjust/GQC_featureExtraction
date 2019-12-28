import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn import decomposition
from pandas.plotting import scatter_matrix
import pandas as pd
import glob
import tensorflow as tf
from tf_random_crop import *

svdnum = 100

def img_inference(x_in):
    rand_crop, offset, size = random_crop(x_in, (28,28,3))
    return rand_crop

def img_load(filename):
    img_raw = tf.io.read_file(filename)
    img = tf.image.decode_image(img_raw)
    offset_width = 50
    offset_height = 10
    target_width = 660 - offset_width
    target_height = 470 - offset_height
    imgc = tf.image.crop_to_bounding_box(img, offset_height, offset_width, target_height, target_width)
    # # args.img_size = 0.25;  args.preserve_aspect_ratio = True; args.rand_box = 0.1
    imresize_ = tf.cast(tf.multiply(tf.cast(imgc.shape[:2], tf.float32), tf.constant(0.25)), tf.int32)
    imgcre = tf.image.resize(imgc, size=imresize_)
    return imgcre

train_data = glob.glob(r'D:\GQC_Images\GQ_Images\Corn_2017_2018/*.png')
train_data = np.vstack([np.expand_dims(img_load(x), axis=0) for x in train_data]) / 255.0
test_data = glob.glob(r'D:\GQC_Images\GQ_Images\test_images_broken/*.png')
test_data = np.vstack([np.expand_dims(img_load(x), axis=0) for x in test_data]) / 255.0
all_data = np.concatenate((train_data, test_data))
# _, _, vh = linalg.svd(data, full_matrices=False)
#
# vh = vh[:,:svdnum]

rand_crops_imgs = []
for _ in range(10):
    temp = [img_inference(x) for x in all_data]
    rand_crops_imgs.extend(temp)
rand_crops_imgs = np.stack(rand_crops_imgs)
rand_crops_imgs_flat = rand_crops_imgs.reshape((rand_crops_imgs.shape[0], -1))

svd = decomposition.TruncatedSVD(n_components=64, n_iter=7, random_state=42)
svd.fit(rand_crops_imgs_flat)

xfm = svd.transform(rand_crops_imgs_flat)
invxfm = svd.inverse_transform(xfm)
invxfm = invxfm.reshape((-1, 28, 28, 3))
# plt.figure();plt.scatter(xfm[:,0], xfm[:,1], c=empty_logs==0)
# plt.figure();plt.scatter(xfm[:,0], xfm[:,1])

# plt.figure();plt.scatter(xfm[empty_logs==0,0], xfm[empty_logs==0,1], alpha=0.1)
# plt.scatter(xfm[empty_logs==1,0], xfm[empty_logs==1,1], alpha=0.1)
#
# df = pd.DataFrame(xfm)
# scatter_matrix(df, alpha = 0.2, figsize = (6, 6), diagonal = 'kde', c=empty_logs==0)