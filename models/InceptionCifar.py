import tensorflow as tf
import tensorflow.contrib.slim as slim


def inception_cifar_arg_scope(weight_decay=0.00004, use_fused_batchnorm=True, training=True):
    batch_norm_params = {
        'is_training': training,
        # Decay for the moving averages.
        'decay': 0.975,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # Use fused batch norm if possible.
        'fused': use_fused_batchnorm,
    }

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params) as sc:
            return sc


def inception_module(net, ch1, ch2, scope='inception_module'):
    with tf.variable_scope(scope):
        net1 = slim.conv2d(net, num_outputs=ch1, kernel_size=(1, 1))
        net2 = slim.conv2d(net, num_outputs=ch2, kernel_size=(3, 3))
        net = tf.concat([net1, net2], axis=3)
        return net


def downsample_module(net, ch1, scope='downsample_module'):
    with tf.variable_scope(scope):
        net1 = slim.conv2d(net, num_outputs=ch1, kernel_size=(3, 3), stride=2)
        net2 = slim.max_pool2d(net, kernel_size=(3, 3), padding='SAME')
        net = tf.concat([net1, net2], axis=3)
        return net


class InceptionCifar:
    def __init__(self, batch_size, im_shape, tag='default'):
        self.name = 'Inception_{}'.format(tag)
        self.batch_size = batch_size
        self.im_shape = im_shape
        self.weight_decay = 0.0005
        self.layers = {}

    def net(self, net, num_classes, reuse=None, training=True, scope='inception'):
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope(inception_cifar_arg_scope(training=training, weight_decay=self.weight_decay)):
                net = slim.conv2d(net, 96, kernel_size=(3, 3), scope='conv_1', padding='VALID')
                self.layers['conv_1'] = net
                net = inception_module(net, 32, 32, scope='incept_1')
                self.layers['incept_1'] = net
                net = inception_module(net, 32, 48, scope='incept_2')
                self.layers['incept_2'] = net
                net = downsample_module(net, 80, scope='down_1')
                self.layers['down_1'] = net

                net = inception_module(net, 112, 48, scope='incept_3')
                self.layers['incept_3'] = net
                net = inception_module(net, 96, 64, scope='incept_4')
                self.layers['incept_4'] = net
                net = inception_module(net, 80, 80, scope='incept_5')
                self.layers['incept_5'] = net
                net = inception_module(net, 48, 96, scope='incept_6')
                self.layers['incept_6'] = net
                net = downsample_module(net, 96, scope='down_2')
                self.layers['down_2'] = net

                net = inception_module(net, 176, 160, scope='incept_7')
                self.layers['incept_7'] = net
                net = inception_module(net, 176, 160, scope='incept_8')
                self.layers['incept_8'] = net
                net = slim.avg_pool2d(net, kernel_size=(7, 7), stride=1)
                net = slim.dropout(net, 0.5, is_training=training)
                net = slim.conv2d(net, num_classes, kernel_size=[1, 1], scope='fc1',
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  biases_initializer=tf.zeros_initializer())
                self.layers['fc_1'] = net
                net = slim.flatten(net)
        return net, self.layers

    def loss(self, scope, preds_train, labels_train, tower=0):
        # Define the loss
        loss = tf.losses.softmax_cross_entropy(labels_train, preds_train, scope=scope)
        tf.summary.scalar('losses/softmax_loss_tower{}'.format(tower), loss)

        # Compute accuracy
        predictions = tf.argmax(preds_train, 1)
        tf.summary.scalar('accuracy/train_accuracy',
                          slim.metrics.accuracy(predictions, tf.argmax(labels_train, 1)))
        tf.summary.histogram('labels', tf.argmax(labels_train, 1))
        tf.summary.histogram('predictions', predictions)
        return loss
