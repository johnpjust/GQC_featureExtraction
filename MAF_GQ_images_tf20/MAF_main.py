import tensorflow as tf
from MAF_GQ_images_tf20 import mafs
import numpy as np
import os
import json
import more_itertools
from munch import munchify

tf.random.set_seed(None)

def load_target_model(path, act_fun = tf.nn.relu):
    # path = r'D:\pycharm_projects\GQC_images_tensorboard\MAF_layers5_h[100]_vhTrue_resize0.25_boxsize0.25_2019-12-27-15-11-42'
    ## set all "args" with json file data
    data = []

    with open(os.path.join(path,'args.json')) as json_file:
        data = json.load(json_file)

    data = [x for x in data.replace("'","").replace(" ","").replace("{", "").replace("}","").split(',') if x.count(":") == 1]
    data = [x.split(":") for x in data]
    data = list(more_itertools.flatten(data))
    data = {item: data[index + 1] for index, item in enumerate(data) if index % 2 == 0}
    for key, value in data.items():
        try:
            data[key] = json.loads(value.lower())
        except:

            data[key] = ''
    args = munchify(data)
    args.act = act_fun ## set manually -- above code doesn't parse function type well
    model = mafs.MaskedAutoregressiveFlow(args.n_dims, args.num_hidden, args.act, args.num_layers, batch_norm=True, args=args)

    model_parms = np.load(os.path.join(path,'model_parms.npy'), allow_pickle=True)

    for m, n in zip(model.parms, model_parms):
        m.assign(n)

    return model