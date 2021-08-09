import argparse
import numpy as np
import hdf5storage
from keras.utils import np_utils
from model.model import BayesOptimizedGNN

if __name__ == '__main__':
    # Parse the command arguments
    args = argparse.ArgumentParser(description='Test a trained Deep Gabor Network for radar classification')
    args.add_argument('-w', '--weight', default='./checkpoints/acc_0.9543_loss_0.0868.h5',
                      type=str,
                      help='Path of the trained weights')
    args.add_argument('-b', '--block', default=3, type=int, help='Number of blocks')
    cmd_args = args.parse_args()

    # Build a Gabor Network
    model = BayesOptimizedGNN(num_block=cmd_args.block)
    model.build_gabor_network()
    model.load_weights(cmd_args.weight)
    print('Loading the weights done')

    # Preprocess the radar data
    test_data = hdf5storage.loadmat('./dataset/smethod/fold1/patches/test.mat')
    test_patches = test_data["test_patches"]
    test_labels = test_data["test_labels"]
    test_patches = np.transpose(test_patches, (2, 0, 1))
    test_patches = np.expand_dims(test_patches, axis=3)
    test_labels = np.squeeze(test_labels, axis=0)
    test_labels = np_utils.to_categorical(test_labels, 3)

    # Evaluate the model
    result = model.evaluate(test_patches, test_labels, verbose=1)
