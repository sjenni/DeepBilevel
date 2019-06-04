import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np
import sys
import os
import time
from datetime import datetime
from utils import remove_missing, get_variables_to_train, montage_tf, get_checkpoint_path, average_gradients
from constants import LOG_DIR

slim = tf.contrib.slim


class BilevelTrainer():
    def __init__(self, model, data_generator, pre_processor, num_epochs, mu=0.01, optimizer='momentum',
                 lr_policy='const', init_lr=0.01, end_lr=None, num_gpus=1, train_scopes='inception', lr_decay=0.95):
        self.model = model
        self.data_generator = data_generator
        self.pre_processor = pre_processor
        self.num_epochs = num_epochs
        self.mu = mu
        self.opt_type = optimizer
        self.lr_policy = lr_policy
        self.init_lr = init_lr
        self.lr_decay = lr_decay
        self.end_lr = end_lr if end_lr is not None else 0.01 * init_lr
        self.num_gpus = num_gpus
        self.num_summary_steps = 80
        self.summaries = []
        self.moving_avgs_decay = 0.9999
        self.global_step = None
        self.var_avg = False
        self.train_scopes = train_scopes
        self.num_train_steps = (self.data_generator.num_train / self.model.batch_size) * self.num_epochs
        self.num_train_steps /= self.num_gpus
        print('Number of training steps: {}'.format(self.num_train_steps))

    def get_data_queue(self):
        dataset = tf.data.Dataset.from_generator(
            lambda: self.data_generator,
            (tf.string, tf.int32),
            (tf.TensorShape([self.model.batch_size]), tf.TensorShape([self.model.batch_size]))
        )
        dataset = dataset.map(self.preprocess, num_parallel_calls=16)
        dataset = dataset.prefetch(buffer_size=5 * self.num_gpus)
        iterator = dataset.make_one_shot_iterator()
        return iterator

    def preprocess(self, filename, label):
        fnames = tf.unstack(filename)
        imgs = []
        for fname in fnames:
            img_string = tf.read_file(fname)
            img = tf.image.decode_png(img_string, channels=3)
            img = self.pre_processor.process_train(img)
            imgs.append(img)

        return tf.stack(imgs, 0), label

    def make_init_fn(self, chpt_path):
        if chpt_path is None:
            return None

        var2restore = slim.get_variables_to_restore(include=[self.train_scopes])
        print('Variables to restore: {}'.format([v.op.name for v in var2restore]))
        var2restore = remove_missing(var2restore, chpt_path)
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(chpt_path, var2restore)
        sys.stdout.flush()

        # Create an initial assignment function.
        def init_fn(sess):
            print('Restoring from: {}'.format(chpt_path))
            sess.run(init_assign_op, init_feed_dict)

        return init_fn

    def build_model(self, batch_queue, tower, opt, scope):
        """
            The main function where the bilevel approach is used
        """
        imgs_train, labels_train = batch_queue.get_next()

        tf.summary.histogram('labels', labels_train)

        # We split the training batches in the pre-defined splits (each containing the same label distribution)
        num_split = self.data_generator.batch_splits
        imgs_train_list = tf.split(imgs_train, num_split)
        labels_train_list = tf.split(labels_train, num_split)

        preds_list = []
        loss_list = []
        # Iterate over all the batch splits
        for i, (imgs, labels) in enumerate(zip(imgs_train_list, labels_train_list)):
            tf.summary.image('imgs/train', montage_tf(imgs, 1, 8), max_outputs=1)

            # Create the model
            reuse = True if (tower > 0 or i > 0) else None
            preds, layers = self.model.net(imgs, self.data_generator.num_classes, reuse=reuse)
            preds_list.append(preds)

            # Compute losses
            loss = self.model.loss(scope, preds, self.data_generator.format_labels(labels), tower)
            tf.get_variable_scope().reuse_variables()

            # Handle dependencies with update_ops (batch-norm)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if update_ops:
                updates = tf.group(*update_ops)
                loss = control_flow_ops.with_dependencies([updates], loss)

            # Store the loss on this split in the list
            loss_list.append(loss)

        # Calculate the gradients on all the batch splits.
        weights = get_variables_to_train(self.train_scopes)
        grads_list = [opt.compute_gradients(l, weights) for l in loss_list]

        # A dictionary with a list of gradients corresponding to the model variables
        grads_accum = {v: [] for (_, v) in grads_list[0]}

        # Flatten the gradients of each split
        grads_flat = [tf.concat([tf.reshape(g, (-1, 1)) for (g, v) in grad], axis=0) for grad in grads_list]

        # Compute the mini-batch weights
        val_grad = grads_flat[0]
        w = [tf.divide(tf.reduce_sum(tf.multiply(val_grad, train_grad)),
                       tf.reduce_sum(tf.multiply(train_grad, train_grad)) + self.mu)
             for train_grad in grads_flat[1:]]

        # Multiply mini-batch gradients by l1 normalized weights
        w_l1norm = tf.reduce_sum(tf.abs(w))
        for i, grads in enumerate(grads_list[1:]):
            for g, v in grads:
                grads_accum[v].append(tf.multiply(g, w[i] / w_l1norm))
        tf.summary.histogram('w', tf.stack(w))

        # Apply weight-decay
        grads_wd = {v: self.model.weight_decay * v if v.op.name.endswith('weights') else 0.0 for (_, v) in
                    grads_list[0]}

        # Accumulate all the gradients per variable
        grads = [(tf.accumulate_n(grads_accum[v]) + grads_wd[v], v) for (_, v) in grads_list[0]]

        self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
        return tf.reduce_mean(loss_list), grads, layers

    def get_save_dir(self):
        fname = '{}_{}'.format(self.model.name, self.data_generator.name)
        return os.path.join(LOG_DIR, '{}/'.format(fname))

    def optimizer(self):
        lr = self.learning_rate()
        opts = {'momentum': tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)}
        return opts[self.opt_type]

    def learning_rate(self):
        policies = {
            'cifar': tf.train.exponential_decay(self.init_lr, self.global_step, self.num_train_steps / self.num_epochs,
                                                self.lr_decay, staircase=True)}
        return policies[self.lr_policy]

    def make_summaries(self, grads, layers):
        self.summaries.append(tf.summary.scalar('learning_rate', self.learning_rate()))
        # Variable summaries
        for variable in slim.get_model_variables():
            self.summaries.append(tf.summary.histogram(variable.op.name, variable))
        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                self.summaries.append(tf.summary.histogram('gradients/' + var.op.name, grad))
        # Add histograms for activation.
        if layers:
            for layer_id, val in layers.iteritems():
                self.summaries.append(tf.summary.histogram('activations/' + layer_id, val))

    def train_model(self, chpt_path):
        print('Restoring from: {}'.format(chpt_path))
        g = tf.Graph()
        with g.as_default():
            with tf.device('/cpu:0'):
                # Init global step
                self.global_step = tf.train.create_global_step()

                batch_queue = self.get_data_queue()
                opt = self.optimizer()

                # Calculate the gradients for each model tower.
                tower_grads = []
                loss = None
                layers = None
                with tf.variable_scope(tf.get_variable_scope()):
                    for i in range(self.num_gpus):
                        with tf.device('/gpu:%d' % i):
                            with tf.name_scope('tower_{}'.format(i)) as scope:
                                loss, grads, layers = self.build_model(batch_queue, i, opt, scope)
                                tower_grads.append(grads)
                grad = average_gradients(tower_grads)

                # Make summaries
                self.make_summaries(grad, layers)

                # Apply the gradients to adjust the shared variables.
                apply_gradient_op = opt.apply_gradients(grad, global_step=self.global_step)

                if self.var_avg:
                    # Track the moving averages of all trainable variables.
                    variable_averages = tf.train.ExponentialMovingAverage(self.moving_avgs_decay, self.global_step)
                    variables_averages_op = variable_averages.apply(tf.trainable_variables())

                    # Group all updates to into a single train op.
                    apply_gradient_op = tf.group(apply_gradient_op, variables_averages_op)

                train_op = control_flow_ops.with_dependencies([apply_gradient_op], loss)

                # Create a saver.
                saver = tf.train.Saver(tf.global_variables())
                init_fn = self.make_init_fn(chpt_path)

                # Build the summary operation from the last tower summaries.
                summary_op = tf.summary.merge(self.summaries)

                # Build an initialization operation to run below.
                init = tf.global_variables_initializer()

                # Start running operations on the Graph.
                sess = tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False), graph=g)
                sess.run(init)
                prev_ckpt = get_checkpoint_path(self.get_save_dir())
                if prev_ckpt:
                    print('Restoring from previous checkpoint: {}'.format(prev_ckpt))
                    saver.restore(sess, prev_ckpt)
                elif init_fn:
                    init_fn(sess)

                summary_writer = tf.summary.FileWriter(self.get_save_dir(), sess.graph)
                init_step = sess.run(self.global_step)
                print('Start training at step: {}'.format(init_step))
                for step in range(init_step, self.num_train_steps):

                    start_time = time.time()
                    _, loss_value = sess.run([train_op, loss])
                    duration = time.time() - start_time

                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    if step % 50 == 0:
                        num_examples_per_step = self.model.batch_size * self.num_gpus
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = duration / self.num_gpus
                        print('{}: step {}/{}, loss = {} ({} examples/sec; {} sec/batch)'
                              .format(datetime.now(), step, self.num_train_steps, loss_value,
                                      examples_per_sec, sec_per_batch))
                        sys.stdout.flush()

                    if step % (self.num_train_steps / self.num_summary_steps) == 0:
                        print('Writing summaries...')
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step)

                    # Save the model checkpoint periodically.
                    if step % (self.num_train_steps / self.num_summary_steps * 4) == 0 or (
                            step + 1) == self.num_train_steps:
                        checkpoint_path = os.path.join(self.get_save_dir(), 'model.ckpt')
                        print('Saving checkpoint to: {}'.format(checkpoint_path))
                        saver.save(sess, checkpoint_path, global_step=step)
