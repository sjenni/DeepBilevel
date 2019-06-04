import tensorflow as tf

slim = tf.contrib.slim


class GenNetTester():
    def __init__(self, model, data_generator, pre_processor):
        self.model = model
        self.pre_processor = pre_processor
        self.data_generator = data_generator
        self.num_eval_steps = (self.data_generator.num_test / self.model.batch_size)

    def get_data_queue(self):
        print('Number of evaluation steps: {}'.format(self.num_eval_steps))

        im_paths, labels = self.data_generator.get_imgpaths_labels()

        im_paths = tf.convert_to_tensor(im_paths, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        data = tf.data.Dataset.from_tensor_slices((im_paths, labels))
        data = data.shuffle(50000)

        data = data.map(self.preprocess, num_parallel_calls=1)

        # create a new dataset with batches of images
        data = data.batch(self.model.batch_size)
        data = data.prefetch(100)
        iterator = data.make_one_shot_iterator()

        return iterator

    def preprocess(self, filename, label):
        img_string = tf.read_file(filename)
        img = tf.image.decode_png(img_string, channels=3)
        img = self.pre_processor.process_test(img)
        return img, label

    def make_test_summaries(self, names_to_values):
        # Create the summary ops such that they also print out to std output:
        summary_ops = []
        for metric_name, metric_value in names_to_values.iteritems():
            op = tf.summary.scalar(metric_name, metric_value)
            op = tf.Print(op, [metric_value], metric_name)
            summary_ops.append(op)
        return summary_ops

    def test_classifier(self, ckpt_dir, tag='default', max_evals=100):
        print('Restoring from: {}'.format(ckpt_dir))

        g = tf.Graph()
        with g.as_default():
            # Get test batches
            batch_queue = self.get_data_queue()
            imgs_test, labels_test = batch_queue.get_next()
            imgs_test.set_shape([self.model.batch_size,]+self.model.im_shape)

            # Get predictions
            predictions, _ = self.model.net(imgs_test, self.data_generator.num_classes, training=False)

            # Compute predicted label for accuracy
            preds_test = tf.argmax(predictions, 1)

            # Choose the metrics to compute:
            names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
                'accuracy_{}'.format(tag): slim.metrics.streaming_accuracy(preds_test, labels_test),
            })
            summary_ops = self.make_test_summaries(names_to_values)

            # Start evaluation
            slim.evaluation.evaluation_loop('', ckpt_dir, ckpt_dir,
                                            num_evals=self.num_eval_steps,
                                            max_number_of_evaluations=max_evals,
                                            eval_op=names_to_updates.values(),
                                            summary_op=tf.summary.merge(summary_ops))
