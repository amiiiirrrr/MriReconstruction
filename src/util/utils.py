"""
utils.py is written to define helper functions
"""

import os
import logging
from time import localtime, strftime
from tensorlayer.prepro import *
import numpy as np
import skimage.measure
import scipy
import tensorflow as tf
import tensorlayer as tl

def distort_img(x_input):
    """
    The distort_img method is written to distort input image
    :param x_input: 3D tensor
    :return x_out: 3D tensor
    """
    x_ss = (x_input + 1.) / 2.
    x_ss = tl.prepro.flip_axis(x_ss, axis=1, is_random=True)
    x_ss = tl.prepro.elastic_transform(x_ss, alpha=255 * 3, sigma=255 * 0.10, is_random=True)
    x_ss = tl.prepro.rotation(x_ss, rg=10, is_random=True, fill_mode='constant')
    x_ss = tl.prepro.shift(x_ss, wrg=0.10, hrg=0.10, is_random=True, fill_mode='constant')
    x_ss = tl.prepro.zoom(x_ss, zoom_range=[0.90, 1.10], is_random=True, fill_mode='constant')
    x_ss = tl.prepro.brightness(x_ss, gamma=0.05, is_random=True)
    x_out = x_ss * 2 - 1
    return x_out

def to_bad_img(x_input, mask):
    """
    The to_bad_img method is written to transfer image to
    fft space then adding with mask
    :param x_input: 3D tensor
    :param mask: 3D tensor
    :return x_out: 3D tensor
    """
    x_ss = (x_input + 1.) / 2.
    fft = scipy.fftpack.fft2(x_ss[:, :, 0])
    fft = scipy.fftpack.fftshift(fft)
    fft = fft * mask
    fft = scipy.fftpack.ifftshift(fft)
    x_ss = scipy.fftpack.ifft2(fft)
    x_ss = np.abs(x_ss)
    x_out = x_ss * 2 - 1
    return x_out[:, :, np.newaxis]

#def correction(out, inputt):
#    out = (out + 1.) / 2.
#    out = scipy.fftpack.fft2(out[:, :, 0])
#    out = scipy.fftpack.fftshift(out)
#    inputt = scipy.fftpack.fft2(scipy.fftpack.fftshift(inputt))
#    for i in range(255):
#        for j in range(255):
#            if inputt[i][j] != 0:
#                out[i][j] = inputt[i][j]
#    out = scipy.fftpack.ifftshift(out)
#    out = scipy.fftpack.ifft2(out)
#    out = np.abs(out)
#    out = out * 2 - 1
#    return out[:, :, np.newaxis]

def fft_abs_for_map_fn(x_input):
    """
    The fft_abs_for_map_fn method is written to calc. fft abs
    :param x_input: 3D tensor
    :param fft_abs: 3D tensor
    """
    x_ss = (x_input + 1.) / 2.
    x_complex = tf.complex(x_ss, tf.zeros_like(x_ss))[:, :, 0]
    fft = tf.spectral.fft2d(x_complex)
    fft_abs = tf.abs(fft)
    return fft_abs

def ssim(data):
    """
    The ssim method is written to calc. similarity between two images
    :param data: 3D tensors
    :param ssim_res: int
    """
    x_good, x_bad = data
    x_good = np.squeeze(x_good)
    x_bad = np.squeeze(x_bad)
    ssim_res = skimage.measure.compare_ssim(x_good, x_bad)
    return ssim_res

def psnr(data):
    """
    The psnr method is written to calc. psnr between two images
    :param data: 3D tensors
    :param psnr_res: int
    """
    x_good, x_bad = data
    psnr_res = skimage.measure.compare_psnr(x_good, x_bad)
    return psnr_res

def vgg_prepro(x_input):
    """
    The vgg_prepro method is written to prepare input for vgg16
    :param x_input: 3D tensors
    :param x_out: 3D tensors
    """
    x_ss = tl.prepro.imresize(x_input, [244, 244], interp='bilinear', mode=None)
    x_ss = np.tile(x_ss, 3)
    x_out = x_ss / 127.5 - 1
    return x_out

def logging_setup(log_dir):
    """
    The logging_setup method is written to prepare logging
    :param log_dir: str
    :param log_all: str
    :param log_eval: str
    :param log_50: str
    :param log_all_filename: str
    :param log_eval_filename: str
    :param log_50_filename: str
    """
    current_time_str = strftime("%Y_%m_%d_%H_%M_%S", localtime())
    log_all_filename = os.path.join(log_dir, 'log_all_{}.log'.format(current_time_str))
    log_eval_filename = os.path.join(log_dir, 'log_eval_{}.log'.format(current_time_str))

    log_all = logging.getLogger('log_all')
    log_all.setLevel(logging.DEBUG)
    log_all.addHandler(logging.FileHandler(log_all_filename))

    log_eval = logging.getLogger('log_eval')
    log_eval.setLevel(logging.INFO)
    log_eval.addHandler(logging.FileHandler(log_eval_filename))

    log_50_filename = os.path.join(log_dir, 'log_50_images_testing_{}.log'.format(current_time_str))

    log_50 = logging.getLogger('log_50')
    log_50.setLevel(logging.DEBUG)
    log_50.addHandler(logging.FileHandler(log_50_filename))

    return log_all, log_eval, log_50, log_all_filename, log_eval_filename, log_50_filename
