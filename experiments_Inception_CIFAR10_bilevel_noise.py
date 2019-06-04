import os
from Preprocessor_cifar import PreprocessorCIFAR
from train.BilevelTrainer import BilevelTrainer
from eval.BilevelTester import GenNetTester
from datasets.Cifar10Generator import CIFAR10Generator
from models.InceptionCifar import InceptionCifar
from constants import CIFAR10_DATADIR


def train_test_model(noise):
    im_shape = [29, 29, 3]
    batch_size = 64
    preprocessor = PreprocessorCIFAR(target_shape=im_shape)

    tag = 'bilevel_noiselevel_{}'.format(noise)

    # Initialize the data generator
    label_file = os.path.join(CIFAR10_DATADIR, 'cifar10_train_noisy_{}.txt'.format(noise))
    data_gen_train_noisy = CIFAR10Generator(label_file, batch_size)

    # Define the network and training
    model_noisy = InceptionCifar(batch_size=batch_size, im_shape=im_shape, tag=tag)
    trainer = BilevelTrainer(model=model_noisy, data_generator=data_gen_train_noisy, pre_processor=preprocessor,
                             num_epochs=100, lr_policy='cifar', num_gpus=1, optimizer='momentum', init_lr=0.1,
                             train_scopes='inception')

    # Train the model
    trainer.train_model(None)
    ckpt = trainer.get_save_dir()

    # Test the model
    label_file = os.path.join(CIFAR10_DATADIR, 'cifar10_test.txt')
    data_generator_test = CIFAR10Generator(label_file, batch_size)
    data_gen_train_noisy.num_test = 50000

    trainer = GenNetTester(model=model_noisy, data_generator=data_generator_test, pre_processor=preprocessor)
    trainer.test_classifier(ckpt, tag='test', max_evals=1)

    trainer = GenNetTester(model=model_noisy, data_generator=data_gen_train_noisy, pre_processor=preprocessor)
    trainer.test_classifier(ckpt, tag='train', max_evals=1)


# Choose the noise levels to train on
noise = [10, 20, 30, 40, 50, 60, 70, 80, 90]
noise = [40]

for s in noise:
    train_test_model(s)
