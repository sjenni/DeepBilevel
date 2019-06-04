import tensorflow as tf

slim = tf.contrib.slim

_PADDING = 4


class PreprocessorCIFAR:
    def __init__(self, target_shape):
        self.target_shape = target_shape

    def process_train(self, image, padding=_PADDING):
        image = tf.to_float(image)
        if padding > 0:
            image = tf.pad(image, [[padding, padding], [padding, padding], [0, 0]])
        # Randomly crop a [height, width] section of the image.
        distorted_image = tf.random_crop(image, self.target_shape)

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, upper=1.8)

        # Subtract off the mean and divide by the variance of the pixels.
        return tf.image.per_image_standardization(distorted_image)

    def process_test(self, image):
        # Transform the image to floats.
        image = tf.to_float(image)

        # Resize and crop if needed.
        resized_image = tf.image.resize_image_with_crop_or_pad(image, self.target_shape[0], self.target_shape[1])

        # Subtract off the mean and divide by the variance of the pixels.
        return tf.image.per_image_standardization(resized_image)
