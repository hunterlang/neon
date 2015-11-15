# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Functions used to load commonly available datasets.
"""

import cPickle
import gzip
import logging
import numpy as np
import os
import sys
import tarfile
import urllib2


names_to_labels = {
'abbey':1,
'airport_terminal':2,
'amphitheater':3,
'amusement_park':4,
'aquarium':5,
'aqueduct':6,
'art_gallery':7,
'assembly_line':8,
'auditorium':9,
'badlands':10,
'bakery':11,
'ballroom':12,
'bamboo_forest':13,
'banquet_hall':14,
'bar':15,
'baseball_field':16,
'bathroom':17,
'beauty_salon':18,
'bedroom':19,
'boat_deck':20,
'bookstore':21,
'botanical_garden':22,
'bowling_alley':23,
'boxing_ring':24,
'bridge':25,
'bus_interior':26,
'butchers_shop':27,
'campsite':28,
'candy_store':29,
'canyon':30,
'cemetery':31,
'chalet':32,
'church':33,
'classroom':34,
'clothing_store':35,
'coast':36,
'cockpit':37,
'coffee_shop':38,
'conference_room':39,
'construction_site':40,
'corn_field':41,
'corridor':42,
'courtyard':43,
'dam':44,
'desert':45,
'dining_room':46,
'driveway':47,
'fire_station':48,
'food_court':49,
'fountain':50,
'gas_station':51,
'golf_course':52,
'harbor':53,
'highway':54,
'hospital_room':55,
'hot_spring':56,
'iceberg':57,
'ice_skating_rink':58,
'kindergarden_classroom':59,
'kitchen':60,
'laundromat':61,
'lighthouse':62,
'living_room':63,
'lobby':64,
'locker_room':65,
'market':66,
'martial_arts_gym':67,
'monastery':68,
'mountain':69,
'museum':70,
'office':71,
'palace':72,
'parking_lot':73,
'phone_booth':74,
'playground':75,
'racecourse':76,
'railroad_track':77,
'rainforest':78,
'restaurant':79,
'river':80,
'rock_arch':81,
'runway':82,
'shed':83,
'shower':84,
'ski_slope':85,
'skyscraper':86,
'slum':87,
'stadium':88,
'stage':89,
'staircase':90,
'subway_station':91,
'supermarket':92,
'swamp':93,
'swimming_pool':94,
'temple':95,
'track':96,
'trench':97,
'valley':98,
'volcano':99,
'yard':100,
}

logger = logging.getLogger(__name__)


def _valid_path_append(path, *args):
    """
    Helper to validate passed path directory and append any subsequent
    filename arguments.

    Arguments:
        path (str): Initial filesystem path.  Should expand to a valid
                    directory.
        *args (list, optional): Any filename or path suffices to append to path
                                for returning.

    Returns:
        (list, str): path prepended list of files from args, or path alone if
                     no args specified.

    Raises:
        ValueError: if path is not a valid directory on this filesystem.
    """
    full_path = os.path.expanduser(path)
    res = []
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    if not os.path.isdir(full_path):
        raise ValueError("path: {0} is not a valid directory".format(path))
    for suffix_path in args:
        res.append(os.path.join(full_path, suffix_path))
    if len(res) == 0:
        return path
    elif len(res) == 1:
        return res[0]
    else:
        return res


def fetch_dataset(url, sourcefile, destfile, totalsz):
    """
    Download the file specified by the given URL.

    Args:
        url (str): Base URL of the file to be downloaded.
        sourcefile (str): Name of the source file.
        destfile (str): Path to the destination.
        totalsz (int): Size of the file to be downloaded.
    """
    cloudfile = urllib2.urlopen(os.path.join(url, sourcefile))
    print("Downloading file: {}".format(destfile))
    blockchar = u'\u2588'  # character to display in progress bar
    with open(destfile, 'wb') as f:
        data_read = 0
        chunksz = 1024**2
        while 1:
            data = cloudfile.read(chunksz)
            if not data:
                break
            data_read = min(totalsz, data_read + chunksz)
            progress_string = u'Download Progress |{:<50}| '.format(
                blockchar * int(float(data_read) / totalsz * 50))
            sys.stdout.write('\r')
            sys.stdout.write(progress_string.encode('utf-8'))
            sys.stdout.flush()

            f.write(data)
        print("Download Complete")


def load_mnist(path=".", normalize=True):
    """
    Fetch the MNIST dataset and load it into memory.

    Args:
        path (str, optional): Local directory in which to cache the raw
                              dataset.  Defaults to current directory.
        normalize (bool, optional): whether to scale values between 0 and 1.
                                    Defaults to True.

    Returns:
        tuple: Both training and test sets are returned.
    """
    mnist = dataset_meta['mnist']
    filepath = _valid_path_append(path, mnist['file'])
    if not os.path.exists(filepath):
        fetch_dataset(mnist['url'], mnist['file'], filepath, mnist['size'])

    with gzip.open(filepath, 'rb') as mnist:
        (X_train, y_train), (X_test, y_test) = cPickle.load(mnist)
        X_train = X_train.reshape(-1, 784)
        X_test = X_test.reshape(-1, 784)

        if normalize:
            X_train = X_train / 255.
            X_test = X_test / 255.

        return (X_train, y_train), (X_test, y_test), 10


def load_places2_mini(normalize=True):
    import cv2 as cv
    labels = []
    count = 0
    for subdir, dirs, files in os.walk("/home/users/hunter/images/train/"):
        for image in files:
            count += 1

    print "found {} images".format(count)
    images = np.ndarray([count, 16384])

    count = 0
    print "loading images"
    for subdir, dirs, files in os.walk("/home/users/hunter/images/train/"):
        for image in files:
            full_path = os.path.join(subdir, image)
            im_arr = cv.imread(full_path);
            im_arr = im_arr.reshape(16384, 3);
            im_arr = im_arr[:,2]
            images[count,:] = im_arr
            label = subdir.split("/")[-1]
            try:
                labels.append(names_to_labels[label]);
            except:
                labels.append(names_to_labels[subdir.split("/")[-2]])
            count+=1
            if count % 10000 == 0:
                print count

    channel_mean = np.mean(images, axis=0)
    channel_mean = channel_mean.reshape(128, 128)
    cv.imwrite("channel_2_mean.jpg", channel_mean)

    print images.shape
    labels = np.asarray(labels)
    labels = labels.reshape(-1, 1)
    print labels.shape

    if normalize:
        im_arr = im_arr / 255.

    return (images, labels)

def load_cifar10(path=".", normalize=True):
    """
    Fetch the CIFAR-10 dataset and load it into memory.

    Args:
        path (str, optional): Local directory in which to cache the raw
                              dataset.  Defaults to current directory.
        normalize (bool, optional): Whether to scale values between 0 and 1.
                                    Defaults to True.

    Returns:
        tuple: Both training and test sets are returned.
    """
    cifar = dataset_meta['cifar-10']
    workdir, filepath = _valid_path_append(path, '', cifar['file'])
    batchdir = os.path.join(workdir, 'cifar-10-batches-py')
    if not os.path.isdir(os.path.join(batchdir, 'data_batch_1')):
        if not os.path.exists(filepath):
            fetch_dataset(cifar['url'], cifar['file'], filepath, cifar['size'])
        with tarfile.open(filepath, 'r:gz') as f:
            f.extractall(workdir)

    train_batches = [os.path.join(batchdir, 'data_batch_' + str(i)) for i in range(1, 6)]
    Xlist, ylist = [], []
    for batch in train_batches:
        with open(batch, 'rb') as f:
            d = cPickle.load(f)
            Xlist.append(d['data'])
            ylist.append(d['labels'])

    X_train = np.vstack(Xlist)
    y_train = np.vstack(ylist)

    with open(os.path.join(batchdir, 'test_batch'), 'rb') as f:
        d = cPickle.load(f)
        X_test, y_test = d['data'], d['labels']

    y_train = y_train.reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    if normalize:
        X_train = X_train / 255.
        X_test = X_test / 255.

    return (X_train, y_train), (X_test, y_test), 10


def load_text(dataset, path="."):
    """
    Fetch the specified dataset.

    Args:
        dataset (str): A key that may be used to retrieve metadata associated
                       with the dataset.
        path (str, optional): Working directory in which to cache loaded data.
                              Defaults to current dir if not specified.

    Returns:
        str: Path to the downloaded dataset.
    """

    text_meta = dataset_meta[dataset]
    workdir, filepath = _valid_path_append(path, '', text_meta['file'])

    if not os.path.exists(filepath):
        fetch_dataset(text_meta['url'], text_meta['file'], filepath,
                      text_meta['size'])
    if '.zip' in filepath:
        import zipfile
        zip_ref = zipfile.ZipFile(filepath)
        zip_ref.extractall(workdir)
        zip_ref.close()
        filepath = filepath.split('.zip')[0]

    return filepath


def load_ptb_train(path):
    return load_text('ptb-train', path)


def load_ptb_valid(path):
    return load_text('ptb-valid', path)


def load_ptb_test(path):
    return load_text('ptb-test', path)


def load_hutter_prize(path):
    return load_text('hutter-prize', path)


def load_shakespeare(path):
    return load_text('shakespeare', path)


def load_flickr8k(path):
    return load_text('flickr8k', path)


def load_flickr30k(path):
    return load_text('flickr30k', path)


def load_coco(path):
    return load_text('coco', path)


def load_i1kmeta(path):
    return load_text('i1kmeta', path)


def load_imdb(path):
    return load_text('imdb', path)


dataset_meta = {
    'mnist': {
        'size': 15296311,
        'file': 'mnist.pkl.gz',
        'url': 'https://s3.amazonaws.com/img-datasets',
        'func': load_mnist
    },
    'cifar-10': {
        'size': 170498071,
        'file': 'cifar-10-python.tar.gz',
        'url': 'http://www.cs.toronto.edu/~kriz',
        'func': load_cifar10
    },
    'ptb-train': {
        'size': 5101618,
        'file': 'ptb.train.txt',
        'url': 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data',
        'func': load_ptb_train
    },
    'ptb-valid': {
        'size': 399782,
        'file': 'ptb.valid.txt',
        'url': 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data',
        'func': load_ptb_valid
    },
    'ptb-test': {
        'size': 449945,
        'file': 'ptb.test.txt',
        'url': 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data',
        'func': load_ptb_test
    },
    'hutter-prize': {
        'size': 35012219,
        'file': 'enwik8.zip',
        'url': 'http://mattmahoney.net/dc',
        'func': load_hutter_prize
    },
    'shakespeare': {
        'size': 4573338,
        'file': 'shakespeare_input.txt',
        'url': 'http://cs.stanford.edu/people/karpathy/char-rnn',
        'func': load_shakespeare
    },
    'flickr8k': {
        'size': 49165563,
        'file': 'flickr8k.zip',
        'url': 'https://s3-us-west-1.amazonaws.com/neon-stockdatasets/image-caption',
        'func': load_flickr8k
    },
    'flickr30k': {
        'size': 195267563,
        'file': 'flickr30k.zip',
        'url': 'https://s3-us-west-1.amazonaws.com/neon-stockdatasets/image-caption',
        'func': load_flickr30k
    },
    'coco': {
        'size': 738051031,
        'file': 'coco.zip',
        'url': 'https://s3-us-west-1.amazonaws.com/neon-stockdatasets/image-caption',
        'func': load_coco
    },
    'i1kmeta': {
        'size': 758648,
        'file': 'neon_ILSVRC2012_devmeta.zip',
        'url': 'https://s3-us-west-1.amazonaws.com/neon-stockdatasets/imagenet',
        'func': load_i1kmeta
    },
    'imdb': {
        'size': 33213513,
        'file': 'imdb.pkl',
        'url': ' https://s3.amazonaws.com/text-datasets',
        'func': load_imdb,
    }
}


def load_dataset(name, path=".", **kwargs):
    """
    Fetch the specified dataset.

    Args:
        name (str): A key that may be used to retrieve the function that
                    can be used to load the dataset.
        path (str, optional): Local cache directory to load the dataset into.
                              Defaults to current working directory.

    Returns:
        tuple: Both training and test sets are returned. The return value
               also contains the number of classes in the dataset.
    """
    if name in dataset_meta:
        if 'func' not in dataset_meta[name]:
            raise ValueError('function not specified for loading %s' % name)
        func = dataset_meta[name]['func']
    else:
        try:
            dataset_module = __import__(name)
        except ImportError:
            raise ValueError('dataset handler not found: %s' % name)
        func = dataset_module.load_data
    return func(path, **kwargs)
