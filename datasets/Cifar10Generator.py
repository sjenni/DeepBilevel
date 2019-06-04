import numpy as np
import os
from random import shuffle
from constants import CIFAR10_DATADIR
import tensorflow as tf

slim = tf.contrib.slim


class CIFAR10Generator:
    def __init__(self, label_file, batch_size, batch_splits=8):
        self.label_file = label_file
        self.data_dir = CIFAR10_DATADIR
        self.name = 'CIFAR10'
        self.num_classes = 10
        self.num_train = 50000
        self.num_test = 10000
        self.label2imgs = {l: [] for l in range(self.num_classes)}
        self.batch_splits = batch_splits
        self.batch_size = batch_size
        self.classes_per_batch = self.batch_size/self.batch_splits
        self.indices = [0]*self.num_classes
        self.num_per_class = [0]*self.num_classes
        self.class_idxs = []
        self.rand_class_idx = 0
        self.setup()

    def __iter__(self):
        while True:
            if self.rand_class_idx+self.classes_per_batch > len(self.class_idxs):
                self.rand_class_idx = 0
            rand_classes = self.class_idxs[self.rand_class_idx:self.rand_class_idx+self.classes_per_batch]
            self.rand_class_idx += self.classes_per_batch
            img_paths = []
            labels = []
            for i in range(self.batch_splits):
                img_paths_, labels_ = self.get_batch_split(rand_classes)
                img_paths += img_paths_
                labels += labels_

            yield img_paths, np.array(labels, dtype=np.int32)

    def get_batch_split(self, rand_classes):
        img_paths = []
        labels = []
        for j in rand_classes:
            img_path = self.label2imgs[j][self.indices[j]]
            self.indices[j] += 1
            if self.indices[j]>self.num_per_class[j]-1:
                self.shuffle_examples(j)
                self.indices[j] = 0
            img_paths.append(img_path)
            labels.append(j)
        return img_paths, labels

    def setup(self):
        print('Initializing Cifar-10 data generator with label file: {}'.format(self.label_file))
        with open(self.label_file, 'r') as l_file:
            lines = [l.strip().split(' ') for l in l_file.readlines()]
            for im_path, label in lines:
                self.label2imgs[int(label)].append(os.path.join(self.data_dir, im_path))
            for c in range(self.num_classes):
                self.num_per_class[c] = len(self.label2imgs[c])
                self.class_idxs += [c]*self.num_per_class[c]
                self.shuffle_examples(c)
            shuffle(self.class_idxs)
            print('Number of images per class: {}'.format(self.num_per_class))
            print('Total number of examples: {}'.format(len(self.class_idxs)))

    def shuffle_examples(self, c):
        shuffle(self.label2imgs[c])

    def format_labels(self, labels):
        return slim.one_hot_encoding(labels, self.num_classes)

    def get_imgpaths_labels(self):
        im_paths_ = []
        labels_ = []
        with open(self.label_file, 'r') as l_file:
            lines = [l.strip().split(' ') for l in l_file.readlines()]
            for im_path, label in lines:
                im_paths_.append(os.path.join(self.data_dir, im_path))
                labels_.append(int(label))
        return im_paths_, labels_