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

def img_preprocessing(x_in, y_in, args):
    # rand_box = np.append(tf.cast(tf.multiply(tf.cast(imgcre.shape[:2], tf.float32),tf.constant(0.1)), tf.int32).numpy(), [3])
    rand_crop, offset, size = random_crop(x_in, args.rand_box)
    # rand_crop = tf.minimum(tf.nn.relu(rand_crop + tf.random.uniform(rand_crop.shape, -0.5, 0.5)), 255)/128.0 - 1  ## dequantize
    # heat_map = xy_in[1][np.int(offset[0] * args.rand_box_ratio):(np.int(offset[0] * args.rand_box_ratio) + args.rand_box_hm_size),
    #            np.int(offset[1] * args.rand_box_ratio):(np.int(offset[1] * args.rand_box_ratio) + args.rand_box_hm_size)]
    heat_map = array_ops.slice(ops.convert_to_tensor(y_in, name="value"), tf.cast(tf.cast(offset[:2], tf.float32) * args.rand_box_ratio,tf.int32), args.rand_box_hm)
    return rand_crop, heat_map

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
    train_data = np.vstack([np.expand_dims(img_load(x, args), axis=0) for x in train_data])/255.0
    test_data = glob.glob(r'D:\GQC_Images\GQ_Images\test_images_broken/*.png')
    test_data = np.vstack([np.expand_dims(img_load(x, args), axis=0) for x in test_data])/255.0

    args.rand_box_size = np.int(train_data[0].shape[0] * args.rand_box_init)
    args.rand_box = np.array([args.rand_box_size, args.rand_box_size, 3])
    args.n_dims = np.prod(args.rand_box)

    train_heatmap = np.load(r'D:\Papers\GQC_images\heatmaps\train_normalized_heatmaps.npy').astype(np.float32)
    test_heatmap = np.load(r'D:\Papers\GQC_images\heatmaps\test_normalized_heatmaps.npy').astype(np.float32)

    args.rand_box_hm_size = np.int(train_heatmap[0].shape[0] * args.rand_box_init)
    args.rand_box_hm = np.array([args.rand_box_hm_size, args.rand_box_hm_size])
    args.rand_box_ratio = args.rand_box_hm_size/args.rand_box_size

    img_preprocessing_ = functools.partial(img_preprocessing, args=args)

    dataset_train = tf.data.Dataset.from_tensor_slices((train_data.astype(np.float32), train_heatmap))  # .float().to(args.device)
    dataset_train = dataset_train.shuffle(buffer_size=len(train_data)).map(img_preprocessing_,
        num_parallel_calls=args.parallel).batch(batch_size=args.batch_dim).prefetch(buffer_size=args.prefetch_size)
    # dataset_train = dataset_train.shuffle(buffer_size=len(train)).batch(batch_size=args.batch_dim).prefetch(buffer_size=args.prefetch_size)

    dataset_valid = tf.data.Dataset.from_tensor_slices((train_data.astype(np.float32), train_heatmap))  # .float().to(args.device)
    dataset_valid = dataset_valid.map(img_preprocessing_, num_parallel_calls=args.parallel).batch(
        batch_size=args.batch_dim * 2).prefetch(buffer_size=args.prefetch_size)
    # dataset_valid = dataset_valid.batch(batch_size=args.batch_dim*2).prefetch(buffer_size=args.prefetch_size)

    dataset_test = tf.data.Dataset.from_tensor_slices((test_data.astype(np.float32), test_heatmap))  # .float().to(args.device)
    dataset_test = dataset_test.map(img_preprocessing_, num_parallel_calls=args.parallel).batch(
        batch_size=args.batch_dim * 2).prefetch(buffer_size=args.prefetch_size)

    return dataset_train, dataset_valid, dataset_test


def create_model(args):
    # random.seed(manualSeed)
    # torch.manual_seed(manualSeed)

    tf.random.set_seed(args.manualSeedw)
    np.random.seed(args.manualSeedw)

    ############## example autoencoder #####################
    # encoder_input = Input(shape=args.rand_box, name='img')
    # x = layers.Conv2D(16, 3, activation='relu')(encoder_input)
    # x = layers.Conv2D(32, 3, activation='relu')(x)
    # x = layers.MaxPooling2D(3)(x)
    # x = layers.Conv2D(32, 3, activation='relu')(x)
    # x = layers.Conv2D(16, 3, activation='relu')(x)
    # encoder_output = layers.GlobalMaxPooling2D()(x)
    #
    # encoder = Model(encoder_input, encoder_output, name='encoder')
    #
    # x = layers.Reshape((4, 4, 1))(encoder_output)
    # x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
    # x = layers.Conv2DTranspose(32, 3, activation='relu')(x)
    # x = layers.UpSampling2D(3)(x)
    # x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
    # decoder_output = layers.Conv2DTranspose(3, 3, activation='relu')(x)
    #
    # autoencoder = Model(encoder_input, decoder_output, name='autoencoder')

    # ######################## resnet autoencoder ############################
    # actfun = tf.nn.relu
    # encoder_input = tf.keras.Input(shape=args.rand_box, name='img')  ## (28, 28, 3) @ 25%/25%
    # x = layers.Conv2D(32, 7, activation=actfun, strides=2)(encoder_input)
    # block_output = layers.AveragePooling2D(3, strides=2)(x)
    #
    # x = layers.Conv2D(32, 1, activation=actfun, padding='same')(block_output)
    # x = layers.Conv2D(32, 3, activation=None, padding='same')(x)
    # x = layers.add([x, block_output])
    # x = layers.Conv2D(16, 1, activation=actfun)(x)
    # # x = layers.Conv2D(16, 1, activation=actfun)(x)
    # # block_output = layers.AveragePooling2D(pool_size=2, strides=2)(x)
    # #
    # # x = layers.Conv2D(32, 1, activation=actfun, padding='same')(block_output)
    # # x = layers.Conv2D(32, 3, activation=None, padding='same')(x)
    # # x = layers.add([x, block_output])
    # # x = layers.Conv2D(32, 1, activation=actfun)(x)
    # # x = layers.AveragePooling2D(2, strides=2)
    # #
    # # x = layers.Conv2D(32, 1, activation=actfun)(block_output)
    # # x = layers.Conv2D(32, 3, activation=None)(x)
    # encoder_output = layers.GlobalAveragePooling2D()(x)
    #
    # encoder = Model(encoder_input, encoder_output, name='encoder')
    #
    # # x = layers.Reshape((2, 2, 1))(encoder_output)
    # # x = layers.Conv2DTranspose(32, 5, activation=actfun, padding='valid')(x)
    # # x = layers.Conv2DTranspose(32, 5, activation=actfun, padding='valid')(x)
    # x = layers.Reshape((4, 4, 1))(encoder_output)
    # x = layers.Conv2DTranspose(32, 7, activation=actfun, padding='valid')(x)
    # block_output = layers.Conv2DTranspose(32, 3, activation=None, padding='valid')(x)
    # x = layers.Conv2DTranspose(32, 1, activation=actfun, padding='same')(block_output)
    # x = layers.Conv2DTranspose(32, 3, activation=None, padding='same')(x)
    # x = layers.add([x, block_output])
    #
    # block_output = x = layers.UpSampling2D(2)(x)
    # x = layers.Conv2DTranspose(32, 1, activation=actfun, padding='same')(block_output)
    # x = layers.Conv2DTranspose(32, 3, activation=None, padding='same')(x)
    # x = layers.add([x, block_output])
    #
    # x = layers.Conv2DTranspose(32, 5, activation=actfun, padding='valid')(x)
    # # x = layers.UpSampling2D(3)(x)
    # decoder_output = tf.squeeze(layers.Conv2DTranspose(3, 3, padding='same', activation=None)(x))
    # # decoder_output = layers.Conv2DTranspose(1, 3,padding='same', activation=actfun)(x)
    # # x = layers.Conv2DTranspose(16, 3, activation=actfun)(x)
    # # x = layers.Conv2DTranspose(32, 3, activation=actfun)(x)
    # # x = layers.UpSampling2D(3)(x)
    # # x = layers.Conv2DTranspose(16, 3, activation=actfun)(x)
    # # decoder_output = layers.Conv2DTranspose(3, 3, activation=actfun)(x)
    #
    # autoencoder = Model(encoder_input, decoder_output, name='autoencoder')

    ######################## dense autoencoder ############################
    actfun = tf.nn.elu
    encoder_input = tf.keras.Input(shape=args.rand_box, name='img')  ## (28, 28, 3) @ 25%/25%
    x = layers.Flatten()(encoder_input)
    x = layers.Dense(512, activation=actfun)(x)
    x = layers.Dense(128, activation=actfun)(x)
    encoder_output = layers.Dense(64, activation=None)(x)

    encoder = Model(encoder_input, encoder_output, name='encoder')

    x = layers.Dense(128, activation=actfun)(encoder_output)
    x = layers.Dense(512, activation=actfun)(x)
    decoder_output = tf.reshape(layers.Dense(28*28*3, activation=None)(x), shape=(-1,28,28,3))

    autoencoder = Model(encoder_input, decoder_output, name='autoencoder')

    ######################## convolutional autoencoder ##########################

    # encoder_input = tf.keras.Input(shape=(28,28,3), name='img')
    # # Encoder Layers
    # x = layers.Conv2D(32, 3, activation='relu', padding='same')(encoder_input)
    # x = layers.MaxPooling2D(2, padding='same')(x)
    # x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    # x = layers.MaxPooling2D(2, padding='same')(x)
    # x = layers.Conv2D(16, 3, strides=2, activation='relu', padding='same')(x)
    #
    # # Flatten encoding for visualization
    # encoder_output = layers.GlobalMaxPooling2D()(x)
    # encoder = Model(encoder_input, encoder_output, name='encoder')
    #
    # # Decoder Layers
    # x = layers.Reshape((4, 4, 1))(encoder_output)
    # x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    # x = layers.UpSampling2D(2)(x)
    # x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    # x = layers.UpSampling2D(2)(x)
    # x = layers.Conv2D(32, 3, activation='relu')(x)
    # x = layers.UpSampling2D(2)(x)
    # decoder_output = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)
    # autoencoder = Model(encoder_input, decoder_output, name='autoencoder')

    # autoencoder.summary()

    ######################### feature extractor autoencoder ################################
    # actfun = tf.nn.relu
    # encoder_input = tf.keras.Input(shape=args.rand_box, name='img') ## (28, 28, 3) @ 25%/25%
    # x = layers.Conv2D(32, 7, activation=actfun, strides=2)(encoder_input)
    # block_output = layers.AveragePooling2D(3, strides=2)(x)
    #
    # x = layers.Conv2D(32, 1, activation=actfun, padding='same')(block_output)
    # x = layers.Conv2D(32, 3, activation=None, padding='same')(x)
    # x = layers.add([x, block_output])
    # x = layers.Conv2D(4, 1, activation=actfun)(x)
    # # x = layers.Conv2D(16, 1, activation=actfun)(x)
    # # block_output = layers.AveragePooling2D(pool_size=2, strides=2)(x)
    # #
    # # x = layers.Conv2D(32, 1, activation=actfun, padding='same')(block_output)
    # # x = layers.Conv2D(32, 3, activation=None, padding='same')(x)
    # # x = layers.add([x, block_output])
    # # x = layers.Conv2D(32, 1, activation=actfun)(x)
    # # x = layers.AveragePooling2D(2, strides=2)
    # #
    # # x = layers.Conv2D(32, 1, activation=actfun)(block_output)
    # # x = layers.Conv2D(32, 3, activation=None)(x)
    # encoder_output = layers.GlobalAveragePooling2D()(x)
    #
    # encoder = Model(encoder_input, encoder_output, name='encoder')
    #
    # x = layers.Reshape((2, 2, 1))(encoder_output)
    # x = layers.Conv2DTranspose(32, 5, activation=actfun, padding='valid')(x)
    # x = layers.Conv2DTranspose(32, 5, activation=actfun, padding='valid')(x)
    # # x = layers.Reshape((4, 4, 1))(encoder_output)
    # x = layers.Conv2DTranspose(32, 7, activation=actfun, padding='valid')(x)
    # block_output = layers.Conv2DTranspose(32, 3, activation=None, padding='valid')(x)
    # x = layers.Conv2DTranspose(32, 1, activation=actfun, padding='same')(block_output)
    # x = layers.Conv2DTranspose(32, 3, activation=None, padding='same')(x)
    # x = layers.add([x, block_output])
    # # x = layers.UpSampling2D(3)(x)
    # decoder_output = tf.squeeze(layers.Conv2DTranspose(1, 3,padding='same', activation=tf.nn.sigmoid)(x))
    # # decoder_output = tf.squeeze(layers.Conv2DTranspose(1, 3,padding='same', activation=actfun)(x))
    # # x = layers.Conv2DTranspose(16, 3, activation=actfun)(x)
    # # x = layers.Conv2DTranspose(32, 3, activation=actfun)(x)
    # # x = layers.UpSampling2D(3)(x)
    # # x = layers.Conv2DTranspose(16, 3, activation=actfun)(x)
    # # decoder_output = layers.Conv2DTranspose(3, 3, activation=actfun)(x)
    #
    # autoencoder = Model(encoder_input, decoder_output, name='autoencoder')

    # x = layers.Flatten()(x)
    # x = layers.Dense(4, activation=actfun)(x)

    encoder.summary()
    autoencoder.summary()

    return autoencoder, encoder


def load_model(args, root, load_start_epoch=False):
    # def f():
    print('Loading model..')
    root.restore(tf.train.latest_checkpoint(args.load or args.path))
    # root.restore(os.path.join(args.load or args.path, 'checkpoint'))
    # if load_start_epoch:
    #     args.start_epoch = tf.train.get_global_step().numpy()
    # return f


def train(model, optimizer, scheduler, data_loader_train, data_loader_val, data_loader_test, args):
    epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):

        for x_mb, y_mb in data_loader_train:
            with tf.GradientTape() as tape:
                # loss = tf.reduce_mean(tf.math.squared_difference(y_mb, model(x_mb, training=True)))
                # loss = tf.reduce_mean(tf.abs(tf.math.subtract(x_mb, model(x_mb, training=True))))
                loss = tf.reduce_mean(tf.math.squared_difference(x_mb, model(x_mb, training=True)))
            grads = tape.gradient(loss, model.trainable_variables)
            grads = [None if grad is None else tf.clip_by_norm(grad, clip_norm=args.clip_norm) for grad in grads]
            globalstep = optimizer.apply_gradients(zip(grads, model.trainable_variables))
            tf.summary.scalar('loss/train', loss, globalstep)

        ## potentially update batch norm variables manually
        ## variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='batch_normalization')

        validation_loss = []
        for x_mb, y_mb in data_loader_val:
            # loss = tf.reduce_mean(tf.math.squared_difference(y_mb, model(x_mb, training=False))).numpy()
            # loss = tf.reduce_mean(tf.abs(tf.math.subtract(x_mb, model(x_mb, training=False)))).numpy()
            loss = tf.reduce_mean(tf.math.squared_difference(x_mb, model(x_mb, training=False))).numpy()
            validation_loss.append(loss)
        validation_loss = tf.reduce_mean(validation_loss)
        # print("validation loss:  " + str(validation_loss))

        test_loss=[]
        for x_mb, y_mb in data_loader_test:
            # loss = tf.reduce_mean(tf.math.squared_difference(y_mb, model(x_mb, training=False))).numpy()
            # loss = tf.reduce_mean(tf.abs(tf.math.subtract(x_mb, model(x_mb, training=False)))).numpy()
            loss = tf.reduce_mean(tf.math.squared_difference(x_mb, model(x_mb, training=False))).numpy()
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


class parser_:
    pass


def main():
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
    args.learning_rate = np.float32(1e-2)
    args.batch_dim = 500
    args.clip_norm = 0.1
    args.epochs = 5000
    args.patience = 10
    args.cooldown = 10
    args.decay = 0.5
    args.min_lr = 5e-4
    args.flows = 6
    args.layers = 1
    args.hidden_dim = 12
    args.residual = 'gated'
    args.expname = ''
    args.load = ''  # r'C:\Users\justjo\PycharmProjects\BNAF_tensorflow_eager\checkpoint\corn_layers1_h12_flows6_resize0.25_boxsize0.1_gated_2019-08-24-11-07-09'
    args.save = True
    args.tensorboard = r'D:\pycharm_projects\GQC_autoencoder'
    args.early_stopping = 50
    args.regL2 = -1
    args.regL1 = -1
    args.manualSeed = None
    args.manualSeedw = None
    args.prefetch_size = 1  # data pipeline prefetch buffer size
    args.parallel = 8  # data pipeline parallel processes
    args.img_size = 0.25;  ## resize img between 0 and 1
    args.preserve_aspect_ratio = True;  ##when resizing
    args.rand_box_init = 0.25  ##relative size of random box from image

    args.path = os.path.join(args.tensorboard,
                             '{}{}_layers{}_h{}_flows{}_resize{}_boxsize{}{}_{}'.format(
                                 args.expname + ('_' if args.expname != '' else ''),
                                 args.dataset, args.layers, args.hidden_dim, args.flows, args.img_size,
                                 args.rand_box_init, '_' + args.residual if args.residual else '',
                                 str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')))

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
        autoencoder, encoder = create_model(args)

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
                               model=autoencoder,
                               optimizer_step=tf.compat.v1.train.get_global_step())

    if args.load:
        load_model(args, root, load_start_epoch=True)

    print('Creating scheduler..')
    # use baseline to avoid saving early on
    scheduler = EarlyStopping(model=autoencoder, patience=args.early_stopping, args=args, root=root)

    with tf.device(args.device):
        train(autoencoder, optimizer, scheduler, data_loader_train, data_loader_valid, data_loader_test, args)

###################### inference #################################
    temp = [x for x in data_loader_train]
    tempae = autoencoder(temp[0][0], training=False)

    train_data = glob.glob(r'D:\GQC_Images\GQ_Images\Corn_2017_2018/*.png')
    train_data = np.vstack([np.expand_dims(img_load(x, args), axis=0) for x in train_data])/255.0
    test_data = glob.glob(r'D:\GQC_Images\GQ_Images\test_images_broken/*.png')
    test_data = np.vstack([np.expand_dims(img_load(x, args), axis=0) for x in test_data])/255.0
    all_data = np.concatenate((train_data, test_data))

    rand_crops_imgs = []
    rand_crops_embeds = []
    for _ in range(10):
        temp = [img_inference(x,args) for x in all_data]
        rand_crops_imgs.extend(temp)
        rand_crops_embeds.extend(encoder(np.stack(temp)))
    rand_crops_imgs = np.stack(rand_crops_imgs)
    rand_crops_embeds = np.stack(rand_crops_embeds)

# if __name__ == '__main__':
#     main()

#### tensorboard --logdir=D:\pycharm_projects\GQC_autoencoder
## http://localhost:6006/