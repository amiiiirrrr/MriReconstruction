"""
run.py is written to train and evaluate the model
"""

import tensorflow as tf
from tensorlayer.prepro import *
import tensorlayer as tl
from tensorlayer.layers import *
from configuration import BaseConfig
from methods import vgg16_cnn_emb, DeepResidualModel
from dataloader import DataLoader
from training import Training
from util import fft_abs_for_map_fn, logging_setup

tf.reset_default_graph()

class Rrunner:
    """
    A class to run the training and testing the model
    """

    def __init__(self):
        """
        initiate Runner class
        """
        self.kwargs = {}
        self.args = BaseConfig().get_args()
        self.basic_config(self.args)
        self.dataloader_obj = DataLoader(self.args)
        self.model_obj = DeepResidualModel()
        self.data_train, self.data_val, self.data_test, self.mask \
            = self.dataloader_obj.data_preparation()
        self.define_model_loss()
        self.training_obj = Training(self.args, self.kwargs)
    def basic_config(self, args):
        """
        BASIC CONFIGS
        :param args: argparser
        :return:
            None
        """
        print('[*] run basic configs ... ')
        log_dir = "log_{}_{}_{}".format(args.model_name, args.mask_name, args.mask_perc)
        tl.files.exists_or_mkdir(log_dir)
        log_all, log_eval, log_50, _, _, _ = logging_setup(log_dir)

        self.kwargs["log_all"] = log_all
        self.kwargs["log_eval"] = log_eval
        self.kwargs["log_50"] = log_50

        checkpoint_dir = "checkpoint_{}_{}_{}".format(
            args.model_name, args.mask_name, args.mask_perc
        )
        tl.files.exists_or_mkdir(checkpoint_dir)

        save_dir = "samples_{}_{}_{}".format(
            args.model_name, args.mask_name, args.mask_perc
        )
        tl.files.exists_or_mkdir(save_dir)

    def define_model_loss(self):
        """
        define model and loss
        :return:
            None
        """
        # ==================================== DEFINE MODEL ==================================== #
        print('[*] define model ... ')

        nwidth, nheight, nzed = self.data_train.shape[1:]

        # define placeholders
        t_image_good = tf.placeholder(
            tf.float32, [self.args.batch_size, nwidth, nheight, nzed], name='good_image'
        )
        t_image_good_samples = tf.placeholder(
            tf.float32, [self.args.sample_size, nwidth, nheight, nzed], name='good_image_samples'
        )
        t_image_bad = tf.placeholder(
            tf.float32, [self.args.batch_size, nwidth, nheight, nzed], name='bad_image'
        )
        t_image_bad_samples = tf.placeholder(
            tf.float32, [self.args.sample_size, nwidth, nheight, nzed], name='bad_image_samples'
        )
        t_gen = tf.placeholder(
            tf.float32, [self.args.batch_size, nwidth, nheight, nzed],
            name='generated_image_for_test'
        )
        t_gen_sample = tf.placeholder(
            tf.float32, [self.args.sample_size, nwidth, nheight, nzed],
            name='generated_sample_image_for_test'
        )
        t_image_good_244 = tf.placeholder(
            tf.float32, [self.args.batch_size, 244, 244, 3], name='vgg_good_image'
        )

        self.kwargs["t_image_good"] = t_image_good
        self.kwargs["t_image_good_samples"] = t_image_good_samples
        self.kwargs["t_image_bad"] = t_image_bad
        self.kwargs["t_image_good_244"] = t_image_good_244
        self.kwargs["t_image_bad_samples"] = t_image_bad_samples
        self.kwargs["t_gen_sample"] = t_gen_sample
        self.kwargs["t_gen"] = t_gen

        self.kwargs["data_train"] = self.data_train
        self.kwargs["data_val"] = self.data_val
        self.kwargs["data_test"] = self.data_test
        self.kwargs["mask"] = self.mask

        # define generator network
        if self.args.model == 'unet':
            # tf.reset_default_graph()
            net = self.model_obj.u_net_bn(
                input_unet=t_image_bad, is_train=True, reuse=False, is_refine=False
            )
            net_test = self.model_obj.u_net_bn(
                input_unet=t_image_bad, is_train=False, reuse=True, is_refine=False
            )
            net_test_sample = self.model_obj.u_net_bn(
                input_unet=t_image_bad_samples, is_train=False, reuse=True, is_refine=False
            )

        elif self.args.model == 'unet_refine':
            net = self.model_obj.u_net_bn(
                input_unet=t_image_bad, is_train=True, reuse=False, is_refine=True
            )
            net_test = self.model_obj.u_net_bn(
                input_unet=t_image_bad, is_train=False, reuse=True, is_refine=True
            )
            net_test_sample = self.model_obj.u_net_bn(
                input_unet=t_image_bad_samples, is_train=False, reuse=True, is_refine=True
            )
        else:
            raise Exception("unknown model")

        self.kwargs["net_test_sample"] = net_test_sample
        self.kwargs["net"] = net
        self.kwargs["net_test"] = net_test

        # define discriminator network
        net_d, logits_fake = self.model_obj.discriminator(
            input_images=net.outputs, is_train=True, reuse=False
        )
        self.kwargs["net_d"] = net_d

        _, logits_real = self.model_obj.discriminator(
            input_images=t_image_good, is_train=True, reuse=True
        )

        # define VGG network
        net_vgg_conv4_good, _ = vgg16_cnn_emb(t_image_good_244, reuse=False)
        # tf.reset_default_graph()
        net_vgg_conv4_gen, _ = vgg16_cnn_emb(
            tf.tile(tf.image.resize_images(
                net.outputs, [244, 244]), [1, 1, 1, 3]), reuse=True
        )

        self.kwargs["net_vgg_conv4_good"] = net_vgg_conv4_good
        # ================================== DEFINE LOSS ================================== #
        print('[*] define loss functions ... ')

        # discriminator loss
        d_loss1 = tl.cost.sigmoid_cross_entropy(
            logits_real, tf.ones_like(logits_real), name='d1'
        )
        d_loss2 = tl.cost.sigmoid_cross_entropy(
            logits_fake, tf.zeros_like(logits_fake), name='d2'
        )
        d_loss = d_loss1 + d_loss2

        self.kwargs["d_loss"] = d_loss

        # generator loss (adversarial)
        g_loss = tl.cost.sigmoid_cross_entropy(
            logits_fake, tf.ones_like(logits_fake), name='g'
        )

        # generator loss (perceptual)
        g_perceptual = tf.reduce_mean(
            tf.reduce_mean(tf.squared_difference(
            net_vgg_conv4_good.outputs,
            net_vgg_conv4_gen.outputs),
            axis=[1, 2, 3])
        )
        self.kwargs["g_perceptual"] = g_perceptual

        # generator loss (pixel-wise)
        g_nmse_a = tf.sqrt(tf.reduce_sum(
            tf.squared_difference(net.outputs, t_image_good), axis=[1, 2, 3])
        )
        g_nmse_b = tf.sqrt(tf.reduce_sum(tf.square(t_image_good), axis=[1, 2, 3]))
        g_nmse = tf.reduce_mean(g_nmse_a / g_nmse_b)
        self.kwargs["g_nmse"] = g_nmse

        # generator loss (frequency)
        fft_good_abs = tf.map_fn(fft_abs_for_map_fn, t_image_good)
        fft_gen_abs = tf.map_fn(fft_abs_for_map_fn, net.outputs)
        g_fft = tf.reduce_mean(tf.reduce_mean(
            tf.squared_difference(fft_good_abs, fft_gen_abs), axis=[1, 2])
        )
        self.kwargs["g_fft"] = g_fft

        # generator loss (total)
        g_loss = self.args.g_adv * g_loss + self.args.g_alpha * g_nmse + \
                 self.args.g_gamma * g_perceptual + self.args.g_beta * g_fft
        self.kwargs["g_loss"] = g_loss

        # nmse metric for testing purpose
        nmse_a_0_1 = tf.sqrt(tf.reduce_sum(
            tf.squared_difference(t_gen, t_image_good), axis=[1, 2, 3])
        )
        nmse_b_0_1 = tf.sqrt(tf.reduce_sum(tf.square(t_image_good), axis=[1, 2, 3]))
        nmse_0_1 = nmse_a_0_1 / nmse_b_0_1
        self.kwargs["nmse_0_1"] = nmse_0_1

        nmse_a_0_1_sample = tf.sqrt(
            tf.reduce_sum(tf.squared_difference(
            t_gen_sample, t_image_good_samples), axis=[1, 2, 3])
        )
        nmse_b_0_1_sample = tf.sqrt(tf.reduce_sum(
            tf.square(t_image_good_samples), axis=[1, 2, 3])
        )
        nmse_0_1_sample = nmse_a_0_1_sample / nmse_b_0_1_sample
        self.kwargs["nmse_0_1_sample"] = nmse_0_1_sample

        # ================================== DEFINE TRAIN OPTS ================================== #
        print('[*] define training options ... ')

        g_vars = tl.layers.get_variables_with_name('u_net', True, True)
        d_vars = tl.layers.get_variables_with_name('discriminator', True, True)

        with tf.variable_scope('learning_rate'):
            lr_v = tf.Variable(self.args.lr, trainable=False)

        g_optim = tf.train.AdamOptimizer(lr_v, beta1=self.args.beta1, beta2=0.999).\
            minimize(g_loss, var_list=g_vars)
        d_optim = tf.train.AdamOptimizer(lr_v, beta1=self.args.beta1, beta2=0.999).\
            minimize(d_loss, var_list=d_vars)

        self.kwargs["d_optim"] = d_optim
        self.kwargs["g_optim"] = g_optim

    def run(self):
        """
        Start training model
        :return:
            None
        """
        tf.reset_default_graph()

        self.training_obj.train()

if __name__ == '__main__':
    run_model_obj = Rrunner()
    run_model_obj.run()
