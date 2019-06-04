import os
from constants import CIFAR10_DATADIR
import numpy as np


def make_cifar10_train_label_file(noise_level=0.5, num_classes=10, num_confusing_classes=9):
    np.random.seed(123)

    clean_label_fname = os.path.join(CIFAR10_DATADIR, 'cifar10_train.txt')
    noisy_label_fname = os.path.join(CIFAR10_DATADIR, 'cifar10_train_noisy_{}.txt'.format(int(noise_level * 100)))

    print('Generating noisy label file in: {}'.format(noisy_label_fname))

    im_paths = []
    labels_clean = []
    labels_dirty = []
    for i in range(num_classes):
        class_dir = 'train/{}'.format(i)
        im_dir = os.path.join(CIFAR10_DATADIR, class_dir)
        imgs = [os.path.join(class_dir, path) for path in os.listdir(im_dir)]
        im_paths += imgs
        clean_labels = np.array([i] * len(imgs))

        # Generate the noisy labels
        random_confusing_classes = np.random.choice(np.concatenate([np.arange(0, i), np.arange(i+1, num_classes)]),
                                                    num_confusing_classes,
                                                    replace=False)
        random_labels = np.random.choice(random_confusing_classes, len(imgs))
        rand_idxs = np.random.choice(np.arange(0, len(imgs)), int(noise_level * len(imgs)), replace=False)
        noisy_labels = np.copy(clean_labels)
        noisy_labels[rand_idxs] = random_labels[rand_idxs]

        labels_clean.append(clean_labels)
        labels_dirty.append(noisy_labels)

    labels_clean = np.concatenate(labels_clean)
    labels_dirty = np.concatenate(labels_dirty)

    assert(np.abs(np.mean(labels_clean == labels_dirty)-(1.-noise_level)) < 1e-3)
    assert(len(labels_clean) == len(im_paths))
    assert(len(labels_clean) == 50000)

    lines_clean = ['{} {}'.format(im_path, label) for im_path, label in zip(im_paths, labels_clean)]
    lines_dirty = ['{} {}'.format(im_path, label) for im_path, label in zip(im_paths, labels_dirty)]

    with open(clean_label_fname, mode='w') as i2l_file:
        i2l_file.write('\n'.join(lines_clean))

    with open(noisy_label_fname, mode='w') as i2l_file:
        i2l_file.write('\n'.join(lines_dirty))


def make_cifar10_test_label_file(num_classes=10):
    clean_label_fname = os.path.join(CIFAR10_DATADIR, 'cifar10_test.txt')

    print('Generating label file in: {}'.format(clean_label_fname))

    im_paths = []
    labels_clean = []
    for i in range(num_classes):
        class_dir = 'test/{}'.format(i)
        im_dir = os.path.join(CIFAR10_DATADIR, class_dir)
        imgs = [os.path.join(class_dir, path) for path in os.listdir(im_dir)]
        im_paths += imgs
        clean_labels = np.array([i] * len(imgs))

        labels_clean.append(clean_labels)

    labels_clean = np.concatenate(labels_clean)

    assert (len(labels_clean) == len(im_paths))
    assert(len(labels_clean) == 10000)

    lines_clean = ['{} {}'.format(im_path, label) for im_path, label in zip(im_paths, labels_clean)]

    with open(clean_label_fname, mode='w') as i2l_file:
        i2l_file.write('\n'.join(lines_clean))