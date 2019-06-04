import os
import numpy as np
import cPickle
import scipy.misc as misc
from constants import CIFAR10_DATADIR
import sys
import urllib
import tarfile

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR10_SRC_DATADIR = os.path.join(CIFAR10_DATADIR, 'cifar-10-batches-py')


def _download_and_extract():
    dest_directory = CIFAR10_DATADIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\rDownloading %s %.2f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print('Downloaded', filename)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def _load_batch(idx=0, load_test=False):
    if load_test:
        batch_file = os.path.join(CIFAR10_SRC_DATADIR, 'test_batch')
    else:
        batch_file = os.path.join(CIFAR10_SRC_DATADIR, 'data_batch_{}'.format(idx))
    with open(batch_file, 'rb') as fo:
        batch = cPickle.load(fo)
    imgs = batch['data']
    labels = batch['labels']
    imgs = [np.transpose(np.reshape(im, (3, 32, 32)), (1, 2, 0)) for im in imgs]
    return imgs, labels


def _init_folder_structure():
    if not os.path.exists(CIFAR10_DATADIR):
        os.makedirs(CIFAR10_DATADIR)

    for setn in ['train', 'test']:
        set_path = os.path.join(CIFAR10_DATADIR, setn)
        if not os.path.exists(set_path):
            os.makedirs(set_path)
        for i in range(10):
            dir_path = '{}/{}'.format(set_path, i)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)


def _write_images(imgs, labels, setn):
    for i, (img, label) in enumerate(zip(imgs, labels)):
        im_path = os.path.join(os.path.join(CIFAR10_DATADIR, setn), '{}/{}.png'.format(label, i))
        misc.imsave(im_path, img)


def convert_cifar10():
    _download_and_extract()
    _init_folder_structure()
    print('Extracting train images to {}'.format(CIFAR10_DATADIR))
    imgs = []
    labels = []
    for i in range(5):
        imgs_, labels_ = _load_batch(i + 1)
        imgs += imgs_
        labels += labels_

    _write_images(imgs, labels, 'train')

    print('Extracting test images to {}'.format(CIFAR10_DATADIR))
    imgs, labels = _load_batch(load_test=True)
    _write_images(imgs, labels, 'test')