import pickle
from pathlib import Path
from functools import reduce

import numpy as np

def load_imagenet(data_dir):

    raise Exception('No Implemented')

    # data_dir = Path(data_dir)
    # data_list = []
    #
    # for path in data_dir.glob('data_batch_*'):
    #
    #     with path.open('rb') as f:
    #
    #         data_list.append(pickle.load(f, encoding='latin1')['data'])
    #
    # with (data_dir / 'test_batch').open('rb') as f:
    #     data_val = pickle.load(f, encoding='latin1')['data']
    #
    # data_len = reduce(lambda x,y:x+y, [d.shape[0] for d in data_list])
    # data_train = np.zeros((data_len, ) + data_list[0].shape[1:])
    # for i, d in enumerate(data_list):
    #     data_train[i*10000:(i+1)*10000] = d
    #
    # data_train = reshape_data(data_train)
    # data_val = reshape_data(data_val)
    #
    # print('Train Data Shape', data_train.shape)
    # print('Val Data Shape', data_val.shape)
    #
    # return (data_train, data_val)

def reshape_data(data):
    data = data.reshape((data.shape[0], 3, 32, 32))
    data = data.transpose((0, 2, 3, 1))
    data = data.astype(np.float32)
    data = data / 255.

    return data
