import os
import json
import datetime
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from lr_scheduler import *
import glob
import random
import pathlib
import scipy
import sklearn.mixture
from tf_random_crop import *
from skimage.transform import resize

from MAF_GQ_images_tf20.MAF_main import *

from tensorflow.keras import layers, Input, Model
import tensorflow as tf

import functools

tf.random.set_seed(None)

def img_inference(x_in, args):
    # rand_box = np.append(tf.cast(tf.multiply(tf.cast(imgcre.shape[:2], tf.float32),tf.constant(0.1)), tf.int32).numpy(), [3])
    rand_crop, offset, size = random_crop(x_in, args.rand_box)
    # rand_crop = tf.minimum(tf.nn.relu(rand_crop + tf.random.uniform(rand_crop.shape, -0.5, 0.5)), 255)/128.0 - 1  ## dequantize
    # heat_map = xy_in[1][np.int(offset[0] * args.rand_box_ratio):(np.int(offset[0] * args.rand_box_ratio) + args.rand_box_hm_size),
    #            np.int(offset[1] * args.rand_box_ratio):(np.int(offset[1] * args.rand_box_ratio) + args.rand_box_hm_size)]
    return rand_crop

def img_preprocessing(x_in, args):
    # rand_box = np.append(tf.cast(tf.multiply(tf.cast(imgcre.shape[:2], tf.float32),tf.constant(0.1)), tf.int32).numpy(), [3])
    rand_crop, offset, size = random_crop(x_in, args.rand_box)
    # rand_crop = tf.clip_by_value(rand_crop + tf.random.uniform(rand_crop.shape, -0.5, 0.5), 0, 255)/128.0 - 1  ## dequantize
    # heat_map = xy_in[1][np.int(offset[0] * args.rand_box_ratio):(np.int(offset[0] * args.rand_box_ratio) + args.rand_box_hm_size),
    #            np.int(offset[1] * args.rand_box_ratio):(np.int(offset[1] * args.rand_box_ratio) + args.rand_box_hm_size)]
    return rand_crop, tf.squeeze(tf.matmul(tf.reshape(rand_crop, [1,-1]), args.vh))

def img_load(filename, args):
    img_raw = tf.io.read_file(filename)
    img = tf.image.decode_image(img_raw)
    offset_width = 50
    offset_height = 10
    target_width = 660 - offset_width
    target_height = 470 - offset_height
    imgc = tf.image.crop_to_bounding_box(img, offset_height, offset_width, target_height, target_width)
    # # args.img_size = 0.25;  args.preserve_aspect_ratio = True; args.rand_box = 0.1
    imresize_ = tf.cast(tf.multiply(tf.cast(imgc.shape[:2], tf.float32), tf.constant(args.img_size)), tf.int32)
    imgcre = tf.image.resize(imgc, size=imresize_)
    return imgcre

def load_dataset(args):
    tf.random.set_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    train_data = glob.glob(r'D:\GQC_Images\GQ_Images\Corn_2017_2018/*.png')
    train_data = np.vstack([np.expand_dims(img_load(x, args), axis=0) for x in train_data])/128.0 - 1
    test_data = glob.glob(r'D:\GQC_Images\GQ_Images\test_images_broken/*.png')
    test_data = np.vstack([np.expand_dims(img_load(x, args), axis=0) for x in test_data])/128.0 - 1

    args.rand_box_size = np.int(train_data[0].shape[0] * args.rand_box_init)
    args.rand_box = np.array([args.rand_box_size, args.rand_box_size, 3])
    args.n_dims = np.prod(args.rand_box)

    img_preprocessing_ = functools.partial(img_preprocessing, args=args)

    dataset_train = tf.data.Dataset.from_tensor_slices(train_data.astype(np.float32))  # .float().to(args.device)
    dataset_train = dataset_train.shuffle(buffer_size=len(train_data)).map(img_preprocessing_,
        num_parallel_calls=args.parallel).batch(batch_size=args.batch_dim).prefetch(buffer_size=args.prefetch_size)
    # dataset_train = dataset_train.shuffle(buffer_size=len(train)).batch(batch_size=args.batch_dim).prefetch(buffer_size=args.prefetch_size)

    dataset_valid = tf.data.Dataset.from_tensor_slices(train_data.astype(np.float32))  # .float().to(args.device)
    dataset_valid = dataset_valid.map(img_preprocessing_, num_parallel_calls=args.parallel).batch(
        batch_size=args.batch_dim * 2).prefetch(buffer_size=args.prefetch_size)
    # dataset_valid = dataset_valid.batch(batch_size=args.batch_dim*2).prefetch(buffer_size=args.prefetch_size)

    dataset_test = tf.data.Dataset.from_tensor_slices(test_data.astype(np.float32))  # .float().to(args.device)
    dataset_test = dataset_test.map(img_preprocessing_, num_parallel_calls=args.parallel).batch(
        batch_size=args.batch_dim * 2).prefetch(buffer_size=args.prefetch_size)

    return dataset_train, dataset_valid, dataset_test


def create_model(args):

    tf.random.set_seed(args.manualSeedw)
    np.random.seed(args.manualSeedw)

    actfun = tf.nn.relu

    inputs = tf.keras.Input(shape=args.rand_box, name='img')  ## (108, 192, 3)
    x = layers.Conv2D(32, 7, activation=actfun, strides=2)(inputs)
    block_output = layers.MaxPooling2D(3, strides=2)(x)

    x = layers.Conv2D(32, 1, activation=actfun, padding='same')(block_output)
    x = layers.Conv2D(32, 3, activation=None, padding='same')(x)
    x = layers.add([x, block_output])
    x = layers.Conv2D(32, 1, activation=actfun)(x)
    # block_output = layers.AveragePooling2D(pool_size=2, strides=2)(x)

    # x = layers.Conv2D(32, 1, activation=actfun, padding='same')(block_output)
    # x = layers.Conv2D(32, 3, activation=None, padding='same')(x)
    # x = layers.add([x, block_output])
    # x = layers.Conv2D(32, 1, activation=actfun)(x)
    # block_output = layers.AveragePooling2D(2, strides=2)(x)

    # x = layers.Conv2D(32, 1, activation=actfun)(block_output)
    # x = layers.Conv2D(32, 3, activation=None)(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(6, activation=tf.nn.elu)(x)
    quality = layers.Dense(1)(x)
    model = tf.keras.Model(inputs, quality, name='toy_resnet')
    model.summary()

    return model


def train(model, optimizer, scheduler, data_loader_train, data_loader_val, data_loader_test, args, target_model):
    epoch = args.start_epoch
    y_mb_scalar = tf.constant(1000, dtype=tf.float32)
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):

        for x_mb, y_mb in data_loader_train:
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(model.trainable_variables)
                loss = tf.reduce_mean(tf.math.squared_difference(target_model.eval(y_mb)/y_mb_scalar, model(x_mb, training=True)))
            grads = tape.gradient(loss, model.trainable_variables)
            grads = [None if grad is None else tf.clip_by_norm(grad, clip_norm=args.clip_norm) for grad in grads]
            globalstep = optimizer.apply_gradients(zip(grads, model.trainable_variables))
            tf.summary.scalar('loss/train', loss, globalstep)

        ## potentially update batch norm variables manually
        ## variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='batch_normalization')

        validation_loss = []
        for x_mb, y_mb in data_loader_val:
            loss = tf.reduce_mean(tf.math.squared_difference(target_model.eval(y_mb)/y_mb_scalar, model(x_mb, training=False))).numpy()
            validation_loss.append(loss)
        validation_loss = tf.reduce_mean(validation_loss)
        # print("validation loss:  " + str(validation_loss))

        test_loss=[]
        for x_mb, y_mb in data_loader_test:
            loss = tf.reduce_mean(tf.math.squared_difference(target_model.eval(y_mb)/y_mb_scalar, model(x_mb, training=False))).numpy()
            test_loss.append(loss)
        test_loss = tf.reduce_mean(test_loss)

        # print("test loss:  " + str(test_loss))

        stop = scheduler.on_epoch_end(epoch=epoch, monitor=validation_loss)

        #### tensorboard
        # tf.summary.scalar('loss/train', train_loss, tf.compat.v1.train.get_global_step())
        tf.summary.scalar('loss/validation', validation_loss, globalstep)
        tf.summary.scalar('loss/test', test_loss, globalstep) ##tf.compat.v1.train.get_global_step()

        if stop:
            break

def load_model(args, root):
    print('Loading model..')
    root.restore(tf.train.latest_checkpoint(args.load or args.path))

class parser_:
    pass


# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# config.log_device_placement = True
# tf.compat.v1.enable_eager_execution(config=config)

# tf.config.experimental_run_functions_eagerly(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

args = parser_()
args.device = '/gpu:0'  # '/gpu:0'
args.dataset = 'corn'  # 'gq_ms_wheat_johnson'#'gq_ms_wheat_johnson' #['gas', 'bsds300', 'hepmass', 'miniboone', 'power']
args.batch_dim = 500
args.clip_norm = 0.1
args.epochs = 5000
args.patience = 10
args.load = r'D:\pycharm_projects\GQC_self_supervised\boxsize0.250.25_2019-12-29-02-05-01'
args.save = True
args.tensorboard = r'D:\pycharm_projects\GQC_self_supervised'
args.early_stopping = 50
args.manualSeed = None
args.manualSeedw = None
args.prefetch_size = 1  # data pipeline prefetch buffer size
args.parallel = 8  # data pipeline parallel processes
args.img_size = 0.25;  ## resize img between 0 and 1 <-- must match target model
args.preserve_aspect_ratio = True;  ##when resizing
args.rand_box_init = 0.25  ##relative size of random box from image <-- must match target model
args.target_model_path = r'D:\pycharm_projects\GQC_images_tensorboard\MAF_layers5_h[100]_vhTrue_resize0.25_boxsize0.25_2019-12-28-14-55-44'

args.path = os.path.join(args.tensorboard,
                         'boxsize{}{}_{}'.format(args.img_size,args.rand_box_init,
                             str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')))

print("loading training/target model")
with tf.device(args.device):
    target_model = load_target_model(args.target_model_path)
args.vh = np.load(os.path.join(args.target_model_path, 'vh.npy'))

print('Loading dataset..')
data_loader_train, data_loader_valid, data_loader_test = load_dataset(args)

if args.save and not args.load:
    print('Creating directory experiment..')
    pathlib.Path(args.path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.path, 'args.json'), 'w') as f:
        json.dump(str(args.__dict__), f, indent=4, sort_keys=True)

# pathlib.Path(args.tensorboard).mkdir(parents=True, exist_ok=True)

print('Creating model..')
with tf.device(args.device):
    model = create_model(args)

## tensorboard and saving
writer = tf.summary.create_file_writer(os.path.join(args.tensorboard, args.load or args.path))
writer.set_as_default()
tf.compat.v1.train.get_or_create_global_step()

global_step = tf.compat.v1.train.get_global_step()
global_step.assign(0)

root = None
args.start_epoch = 0

print('Creating optimizer..')
with tf.device(args.device):
    optimizer = tf.optimizers.Adam()
root = tf.train.Checkpoint(optimizer=optimizer,
                           model=model,
                           optimizer_step=tf.compat.v1.train.get_global_step())

if args.load:
    load_model(args, root)

print('Creating scheduler..')
# use baseline to avoid saving early on
scheduler = EarlyStopping(model=model, patience=args.early_stopping, args=args, root=root)

with tf.device(args.device):
    train(model, optimizer, scheduler, data_loader_train, data_loader_valid, data_loader_test, args, target_model)

# ###################### inference #################################
    embeds = tf.keras.Model(model.input, model.layers[-2].output, name='embeds')

    # train_data = glob.glob(r'D:\GQC_Images\GQ_Images\Corn_2017_2018/*.png')
    # train_data = np.vstack([np.expand_dims(img_load(x, args), axis=0) for x in train_data])/128.0 - 1
    # test_data = glob.glob(r'D:\GQC_Images\GQ_Images\test_images_broken/*.png')
    # test_data = np.vstack([np.expand_dims(img_load(x, args), axis=0) for x in test_data])/128.0 - 1
    # all_data = np.concatenate((train_data, test_data))

    rand_crops_imgs = []
    rand_crops_embeds = []
    for _ in range(10):
        temp = [x for x in data_loader_train]
        rand_crops_imgs.extend(temp[0][0].numpy())
        rand_crops_embeds.extend(embeds(temp[0][0]))

    for _ in range(10):
        temp = [x for x in data_loader_test]
        rand_crops_imgs.extend(temp[0][0].numpy())
        rand_crops_embeds.extend(embeds(temp[0][0]))

    rand_crops_imgs = np.stack(rand_crops_imgs)
    rand_crops_embeds = np.stack(rand_crops_embeds)

# if __name__ == '__main__':
#     main()

#### tensorboard --logdir=D:\pycharm_projects\GQC_self_supervised
## http://localhost:6006/