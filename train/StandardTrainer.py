import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from utils import get_variables_to_train, montage_tf

from BilevelTrainer import BilevelTrainer

slim = tf.contrib.slim


class StandardTrainer(BilevelTrainer):
    def __init__(self, *args, **kwargs):
        BilevelTrainer.__init__(self, *args, **kwargs)

    def build_model(self, batch_queue, tower, opt, scope):
        imgs_train, labels_train = batch_queue.get_next()

        tf.summary.image('imgs/train', montage_tf(imgs_train, 1, 8), max_outputs=1)

        # Create the model
        reuse = True if (tower > 0) else None
        preds, layers = self.model.net(imgs_train, self.data_generator.num_classes, reuse=reuse)

        # Compute losses
        loss = self.model.loss(scope, preds, self.data_generator.format_labels(labels_train), tower)
        tf.get_variable_scope().reuse_variables()

        # Handle dependencies with update_ops (batch-norm)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            loss = control_flow_ops.with_dependencies([updates], loss)

        # Calculate the gradients on all the batch splits.
        weights = get_variables_to_train(self.train_scopes)
        grads = opt.compute_gradients(loss, weights)

        # Apply weight-decay
        grads_wd = {v: self.model.weight_decay * v if v.op.name.endswith('weights') else 0.0 for (_, v) in
                    grads}

        # Accumulate all the gradients per variable
        grads = [(g + grads_wd[v], v) for (g, v) in grads]

        self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
        return loss, grads, layers
