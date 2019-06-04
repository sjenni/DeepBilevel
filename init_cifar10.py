from datasets import convert_cifar10, cifar10_label_files

convert_cifar10.convert_cifar10()

noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for noise_level in noise_levels:
    cifar10_label_files.make_cifar10_train_label_file(noise_level)

cifar10_label_files.make_cifar10_test_label_file()