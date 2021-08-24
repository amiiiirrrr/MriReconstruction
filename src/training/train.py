"""
training.py is written to train and evaluate the model
"""

import time
import os
import tensorflow as tf
import numpy as np
from tensorlayer.prepro import *
import tensorlayer as tl
from tensorlayer.layers import *
from scipy.io import savemat
import scipy.misc as misc
from ..util import distort_img, to_bad_img, \
    ssim, psnr, vgg_prepro

tf.reset_default_graph()

class Training:
    """
    The Training class is written to train and evaluate the GAN model
    """
    def __init__(self, args, kwargs):
        """
        The initiate method
        :param args: argparser
        :return:
            None
        """
        self.args = args
        self.kwargs = kwargs
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        tl.layers.initialize_global_variables(self.sess)

    def train(self):
        """
        The train method is written to train GAN model
        :return:
            None
        """
        tl.files.load_and_assign_npz(sess=self.sess,
                                     name=os.path.join(
                                     self.args.checkpoint_dir, tl.global_flag['model']) + '.npz',
                                     network=self.kwargs["net"]
                                     )
        tl.files.load_and_assign_npz(sess=self.sess,
                                     name=os.path.join(
                                     self.args.checkpoint_dir, tl.global_flag['model']) + '_d.npz',
                                     network=self.kwargs["net_d"]
                                     )
        # load vgg weights
        net_vgg_conv4_path = self.args.VGG16_path
        npz = np.load(net_vgg_conv4_path)
        assign_op = []
        for idx, val in enumerate(sorted(npz.items())[0:20]):
            print("  Loading pretrained VGG16, CNN part %s" % str(val[1].shape))
            assign_op.append(self.kwargs["net_vgg_conv4_good"].all_params[idx].assign(val[1]))
        self.sess.run(assign_op)
        self.kwargs["net_vgg_conv4_good"].print_params(False)

        n_training_examples = len(self.kwargs["data_train"])
        n_step_epoch = round(n_training_examples / self.args.batch_size)

        # sample testing images
        idex = tl.utils.get_random_int(
            min_v=0, max_v=len(self.kwargs["data_test"]) - 1,
            number=self.args.sample_size, seed=self.args.seed
        )
        x_samples_good = self.kwargs["data_test"][idex]
        x_samples_bad = tl.prepro.threading_data(
            x_samples_good, fn=to_bad_img, mask=self.kwargs["mask"]
        )
        x_good_sample_rescaled = (x_samples_good + 1) / 2
        x_good_sample_rescaled = x_good_sample_rescaled.astype(np.float32)
        x_bad_sample_rescaled = (x_samples_bad + 1) / 2
        x_bad_sample_rescaled = x_bad_sample_rescaled.astype(np.float32)

        self.visualize(x_samples_good, x_samples_bad, self.kwargs["mask"])

        print('[*] start training ... ')

        best_nmse = np.inf
        best_epoch = 1
        esn = self.args.early_stopping_num
        for epoch in range(0, self.args.n_epoch):

            # learning rate decay
            if epoch != 0 and (epoch % self.args.decay_every == 0):
                new_lr_decay = self.args.lr_decay ** (epoch // self.args.decay_every)
                self.sess.run(tf.assign(self.args.lr_v, self.args.lr * new_lr_decay))
                log = " ** new learning rate: %f" % (self.args.lr * new_lr_decay)
                print(log)
                self.kwargs["log_all"].debug(log)
            elif epoch == 0:
                log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (
                    self.args.lr, self.args.decay_every, self.args.lr_decay
                )
                print(log)
                self.kwargs["log_all"].debug(log)

            for step in range(n_step_epoch):
                step_time = time.time()
                idex = tl.utils.get_random_int(
                    min_v=0, max_v=n_training_examples - 1, number=self.args.batch_size
                )
                x_good = self.kwargs["data_train"][idex]
                x_good_aug = tl.prepro.threading_data(x_good, fn=distort_img)
                x_good_244 = tl.prepro.threading_data(x_good_aug, fn=vgg_prepro)
                x_bad = tl.prepro.threading_data(
                    x_good_aug, fn=to_bad_img, mask=self.kwargs["mask"]
                )

                err_dis, _ = self.sess.run(
                    [self.kwargs["d_loss"], self.kwargs["d_optim"]],
                    {self.kwargs["t_image_good"]: x_good_aug, self.kwargs["t_image_bad"]: x_bad}
                )
                err_gen, err_gen_perceptual, err_gen_nmse, err_gen_fft, _ = self.sess.run(
                    [self.kwargs["g_loss"], self.kwargs["g_perceptual"],
                    self.kwargs["g_nmse"], self.kwargs["g_fft"], self.kwargs["g_optim"]],
                    {self.kwargs["t_image_good_244"]: x_good_244,
                    self.kwargs["t_image_good"]: x_good_aug,
                    self.kwargs["t_image_bad"]: x_bad}
                )

                log = \
                    "Epoch[{:3}/{:3}] step={:3} d_loss={:5} g_loss={:5}" \
                    " g_perceptual_loss={:5} g_mse={:5} g_freq={:5} took {:3}s".format(
                    epoch + 1,
                    self.args.n_epoch,
                    step,
                    round(float(err_dis), 3),
                    round(float(err_gen), 3),
                    round(float(err_gen_perceptual), 3),
                    round(float(err_gen_nmse), 3),
                    round(float(err_gen_fft), 3),
                    round(time.time() - step_time, 2))
                print(log)
                self.kwargs["log_all"].debug(log)
                #if(err_gen<1.5):
                #    break

            # evaluation for training data
            _ = self.evaluation(self.kwargs["data_train"])

            # evaluation for validation data
            total_nmse_val = self.evaluation(self.kwargs["data_val"])


            img = self.sess.run(
                self.kwargs["net_test_sample"].outputs,
                {self.kwargs["t_image_bad_samples"]: x_samples_bad}
            )
            tl.visualize.save_images(img,
                                     [5, 10],
                                     os.path.join(self.args.save_dir, "image_{}.png".format(epoch))
                                     )

            if total_nmse_val < best_nmse:
                esn = self.args.early_stopping_num  # reset early stopping num
                best_nmse = total_nmse_val
                best_epoch = epoch + 1

                # save current best model
                tl.files.save_npz(self.kwargs["net"].all_params,
                                  name=os.path.join(
                                  self.args.checkpoint_dir,
                                  tl.global_flag['model']) + '.npz', sess=self.sess
                                  )

                tl.files.save_npz(self.kwargs["net_d"].all_params,
                                    name=os.path.join(self.args.checkpoint_dir,
                                    tl.global_flag['model']) + '_d.npz',
                                    sess=self.sess)

                print("[*] Save checkpoints SUCCESS!")
            else:
                esn -= 1

            log = "Best NMSE result: {} at {} epoch".format(best_nmse, best_epoch)
            self.kwargs["log_eval"].info(log)
            self.kwargs["log_all"].debug(log)
            print(log)

            # early stopping triggered
            if esn == 0:
                self.kwargs["log_eval"].info(log)

                tl.files.load_and_assign_npz(
                    sess=self.sess,
                    name=os.path.join(self.args.checkpoint_dir, tl.global_flag['model']) + '.npz',
                    network=self.kwargs["net"])
                # evluation for test data
                x_gen = self.sess.run(
                    self.kwargs["net_test_sample"].outputs,
                    {self.kwargs["t_image_bad_samples"]: x_samples_bad}
                )
                x_gen_0_1 = (x_gen + 1) / 2
                savemat(
                    self.args.save_dir + '/test_random_50_generated.mat', {'x_gen_0_1': x_gen_0_1}
                )

                nmse_res = self.sess.run(
                    self.kwargs["nmse_0_1_sample"],
                    {self.kwargs["t_gen_sample"]: x_gen_0_1,
                     self.kwargs["t_image_good_samples"]: x_good_sample_rescaled}
                )
                ssim_res = tl.prepro.threading_data(
                    [_ for _ in zip(x_good_sample_rescaled, x_gen_0_1)], fn=ssim
                )
                psnr_res = tl.prepro.threading_data(
                    [_ for _ in zip(x_good_sample_rescaled, x_gen_0_1)], fn=psnr
                )

                log = "NMSE testing: {}\nSSIM testing: {}\nPSNR testing: {}\n\n".format(
                    nmse_res,
                    ssim_res,
                    psnr_res)

                self.kwargs["log_50"].debug(log)

                log = "NMSE testing average: {}\nSSIM testing average: " \
                      "{}\nPSNR testing average: {}\n\n".format(
                    np.mean(nmse_res),
                    np.mean(ssim_res),
                    np.mean(psnr_res)
                )

                self.kwargs["log_50"].debug(log)

                log = "NMSE testing std: {}\nSSIM testing std: " \
                      "{}\nPSNR testing std: {}\n\n".format(np.std(
                        nmse_res),
                        np.std(ssim_res),
                        np.std(psnr_res)
                )

                self.kwargs["log_50"].debug(log)

                # evaluation for zero-filled (ZF) data
                nmse_res_zf = self.sess.run(
                    self.kwargs["nmse_0_1_sample"],
                    {
                        self.kwargs["t_gen_sample"]: x_bad_sample_rescaled,
                        self.kwargs["t_image_good_samples"]: x_good_sample_rescaled
                    }
                )
                ssim_res_zf = tl.prepro.threading_data(
                    [_ for _ in zip(x_good_sample_rescaled, x_bad_sample_rescaled)], fn=ssim
                )
                psnr_res_zf = tl.prepro.threading_data(
                    [_ for _ in zip(x_good_sample_rescaled, x_bad_sample_rescaled)], fn=psnr
                )

                log = "NMSE ZF testing: {}\nSSIM ZF testing: {}\nPSNR ZF testing: {}\n\n".format(
                    nmse_res_zf,
                    ssim_res_zf,
                    psnr_res_zf
                )

                self.kwargs["log_50"].debug(log)

                log = "NMSE ZF average testing: {}\nSSIM ZF average testing: " \
                      "{}\nPSNR ZF average testing: {}\n\n".format(
                    np.mean(nmse_res_zf),
                    np.mean(ssim_res_zf),
                    np.mean(psnr_res_zf)
                )

                self.kwargs["log_50"].debug(log)

                log = "NMSE ZF std testing: {}\nSSIM ZF std testing: " \
                      "{}\nPSNR ZF std testing: {}\n\n".format(
                    np.std(nmse_res_zf),
                    np.std(ssim_res_zf),
                    np.std(psnr_res_zf)
                )

                self.kwargs["log_50"].debug(log)

                # sample testing images
                tl.visualize.save_images(x_gen,
                                         [5, 10],
                                         os.path.join(
                                             self.args.save_dir, "final_generated_image.png"
                                         )
                                    )

                tl.visualize.save_images(np.clip(
                    10 * np.abs(x_samples_good - x_gen) / 2, 0, 1),
                    [5, 10],
                    os.path.join(self.args.save_dir, "final_generated_image_diff_abs_10_clip.png"))

                tl.visualize.save_images(np.clip(
                    10 * np.abs(x_samples_good - x_samples_bad) / 2, 0, 1),
                    [5, 10],
                    os.path.join(self.args.save_dir, "final_bad_image_diff_abs_10_clip.png"))

                print("[*] Job finished!")
                break

    def evaluation(self, data_iter):
        """
        The evaluation method is written to evaluate GAN model
        :param data_iter: data
        :return total_nmse: float
        """
        total_nmse = 0
        total_ssim = 0
        total_psnr = 0
        num_temp = 0
        for batch in tl.iterate.minibatches(inputs=data_iter, targets=data_iter,
                                            batch_size=self.args.batch_size, shuffle=False):
            x_good, _ = batch
            # x_bad = threading_data(x_good, fn=to_bad_img, mask=self.kwargs["mask"])
            x_bad = tl.prepro.threading_data(
                x_good,
                fn=to_bad_img,
                mask=self.kwargs["mask"])

            x_gen = self.sess.run(self.kwargs["net_test"].outputs,
                                  {self.kwargs["t_image_bad"]: x_bad})

            x_good_0_1 = (x_good + 1) / 2
            x_gen_0_1 = (x_gen + 1) / 2
            x_good_0_1 = x_good_0_1.astype(np.float32)
            nmse_res = self.sess.run(
                self.kwargs["nmse_0_1"],
                {self.kwargs["t_gen"]: x_gen_0_1, self.kwargs["t_image_good"]: x_good_0_1}
            )
            ssim_res = tl.prepro.threading_data([_ for _ in zip(x_good_0_1, x_gen_0_1)], fn=ssim)
            psnr_res = tl.prepro.threading_data([_ for _ in zip(x_good_0_1, x_gen_0_1)], fn=psnr)
            total_nmse += np.sum(nmse_res)
            total_ssim += np.sum(ssim_res)
            total_psnr += np.sum(psnr_res)
            num_temp += self.args.batch_size

        total_nmse /= num_temp
        total_ssim /= num_temp
        total_psnr /= num_temp

        log = "Epoch: {}\nNMSE training: {:8}, SSIM training: {:8}, PSNR training: {:8}".format(
            self.args.epoch + 1,
            total_nmse,
            total_ssim,
            total_psnr)
        print(log)
        self.kwargs["log_all"].debug(log)
        self.kwargs["log_eval"].info(log)

        return total_nmse

    def visualize(self, x_samples_good, x_samples_bad, mask):
        """
        The visualize method is written to visualize samples
        :param x_samples_good: 4D Tensor
        :param x_samples_bad: 4D Tensor
        :param mask: 3D Tensor
        :return:
            None
        """
        tl.visualize.save_images(x_samples_good,
                                 [5, 10],
                                 os.path.join(self.args.save_dir, "sample_image_good.png"))

        tl.visualize.save_images(x_samples_bad,
                                 [5, 10],
                                 os.path.join(self.args.save_dir, "sample_image_bad.png"))

        tl.visualize.save_images(np.abs(x_samples_good - x_samples_bad),
                                 [5, 10],
                                 os.path.join(self.args.save_dir, "sample_image_diff_abs.png"))

        tl.visualize.save_images(np.sqrt(
            np.abs(x_samples_good - x_samples_bad) / 2 + self.args.epsilon),
             [5, 10],
             os.path.join(self.args.save_dir, "sample_image_diff_sqrt_abs.png"))

        tl.visualize.save_images(np.clip(10 * np.abs(x_samples_good - x_samples_bad) / 2, 0, 1),
                                [5, 10],
                                os.path.join(
                                self.args.save_dir, "sample_image_diff_sqrt_abs_10_clip.png")
                                 )

        tl.visualize.save_images(tl.prepro.threading_data(x_samples_good, fn=distort_img),
                                 [5, 10],
                                 os.path.join(self.args.save_dir, "sample_image_aug.png"))
        misc.imsave(os.path.join(self.args.save_dir, "mask.png"),mask * 255)
