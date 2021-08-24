"""
data_loader.py is written to load dataset
"""

import os
import pickle
import tensorlayer as tl
import numpy as np
from PIL import Image
from scipy.io import loadmat, savemat

class DataLoader:
    """
    class for the load dataset
    """
    def __init__(self, args):
        """
        The get_args method is written to return config arguments
        :return: dict
        """
        self.args = args
        self.x_train = []
        self.x_test = []
        self.x_val = []
        self.save_datasets()

    def train_loader(self):
        """
        The train_loader method is written to load train data
        :return X_train: list
        """
        for path in os.listdir(self.args.training_data_path):
            img_path = os.path.join(self.args.training_data_path, path)
            img = Image.open(img_path)
            img_3d_max = np.max(img)
            img = img / img_3d_max * 255
            img = img / 127.5 - 1
            img_2d = np.transpose(img, (1, 0))
            a_axis = img_2d.shape[0]
            b_axis = img_2d.shape[1]
            if a_axis == 256 & b_axis == 256 :
                self.x_train.append(img_2d)

    def test_loader(self):
        """
        The test_loader method is written to load test data
        :return X_test: list
        """
        for path in os.listdir(self.args.testing_data_path):
            img_path = os.path.join(self.args.testing_data_path,path)
            img = Image.open(img_path)
            img_3d_max = np.max(img)
            img = img / img_3d_max * 255
            img = img / 127.5 - 1
            img_2d = np.transpose(img, (1, 0))
            a_axis = img_2d.shape[0]
            b_axis = img_2d.shape[1]
            if a_axis == 256 & b_axis == 256 :
                self.x_test.append(img_2d)

    def valid_loader(self):
        """
        The valid_loader method is written to load validation data
        :return X_val: list
        """

        for path in os.listdir(self.args.validation_data_path):
            img_path = os.path.join(self.args.validation_data_path,path)
            img = Image.open(img_path)
            img_3d_max = np.max(img)
            img = img / img_3d_max * 255
            img = img / 127.5 - 1
            img_2d = np.transpose(img, (1, 0))
            a_axis = img_2d.shape[0]
            b_axis = img_2d.shape[1]
            if a_axis == 256 & b_axis == 256 :
                self.x_val.append(img_2d)

    def save_datasets(self):
        """
        The save_datasets method is written to save datasets as pickle format
        :return:
            None
        """
        self.train_loader()
        self.test_loader()
        self.valid_loader()

        x_train = np.asarray(self.x_train)
        x_train = x_train[:, :, :, np.newaxis]
        x_val = np.asarray(self.x_val)
        x_val = x_val[:, :, :, np.newaxis]
        x_test = np.asarray(self.x_test)
        x_test = x_test[:, :, :, np.newaxis]
        #X_train = X_train.astype(np.float32)
        #X_val = X_val.astype(np.float32)
        #X_test = X_test.astype(np.float32)
        # save data into pickle format

        tl.files.exists_or_mkdir(self.args.data_saving_path)

        print("save training set into pickle format")
        with open(os.path.join(
                self.args.data_saving_path, 'training.pickle'), 'wb') as f_train:
            pickle.dump(x_train, f_train, protocol=4)

        print("save validation set into pickle format")
        with open(os.path.join(
                self.args.data_saving_path, 'validation.pickle'), 'wb') as f_valid:
            pickle.dump(x_val, f_valid, protocol=4)

        print("save test set into pickle format")
        with open(os.path.join(
                self.args.data_saving_path, 'testing.pickle'), 'wb') as f_test:
            pickle.dump(x_test, f_test, protocol=4)

        print("processing data finished!")

    def data_preparation(self):
        """
        data preparation
        :return data_train: pickle
        :return data_val: pickle
        :return data_test: pickle
        :return mask: 3D Tensor
        """
        print('[*] load data ... ')
        training_data_path = self.args.training_data_path
        val_data_path = self.args.val_data_path
        testing_data_path = self.args.testing_data_path

        with open(training_data_path, 'rb') as f:
            data_train = pickle.load(f)

        with open(val_data_path, 'rb') as f:
            data_val = pickle.load(f)

        with open(testing_data_path, 'rb') as f:
            data_test = pickle.load(f)

        print('X_train shape/min/max: ', data_train.shape, data_train.min(), data_train.max())
        print('X_val shape/min/max: ', data_val.shape, data_val.min(), data_val.max())
        print('X_test shape/min/max: ', data_test.shape, data_test.min(), data_test.max())

        print('[*] loading mask ... ')
        if self.args.mask == "gaussian2d":
            mask = \
                loadmat(
                    os.path.join(
                        self.args.TRAIN.mask_Gaussian2D_path,
                        "GaussianDistribution2DMask_{}.mat".format(
                            self.args.maskperc
                        )))['maskRS2']
        elif self.args.mask == "gaussian1d":
            mask = \
                loadmat(
                    os.path.join(
                        self.args.TRAIN.mask_Gaussian1D_path,
                        "GaussianDistribution1DMask_{}.mat".format(
                        self.args.maskperc
                        )))['maskRS1']
        elif self.args.mask == "radial2d":
            mask = \
                loadmat(
                    os.path.join(
                        self.args.TRAIN.mask_Radial2D_path,
                        "RadialDistributionMask_{}.mat".format(
                            self.args.maskperc
                        )))['img']
        else:
            raise ValueError("no such mask exists: {}".format(self.args.mask))

        return data_train, data_val, data_test, mask