"""
vgg16.py is written to define vgg16 model
"""

import tensorflow as tf
import tensorlayer as tl

def vgg16_cnn_emb(t_image, reuse=False):
    """
    The vgg16_cnn_emb method is written to define vgg16 model
    :param t_image: 4D tensor
    :param reuse: bool
    :return conv4: 4D tensor
    :return network: 3D tensor
    """
    with tf.variable_scope("vgg16_cnn", reuse=reuse) as vss:
        tl.layers.set_name_reuse(reuse)
        t_image = (t_image + 1) * 127.5  # convert input of [-1, 1] to [0, 255]

        mean = tf.constant(
            [123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        net_in = tl.layers.InputLayer(t_image - mean, name='vgg_input_im')

        # conv1
        network = tl.layers.Conv2dLayer(net_in,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 3, 64],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv1_1')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 64, 64],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv1_2')
        network = tl.layers.PoolLayer(network,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      pool=tf.nn.max_pool,
                                      name='vgg_pool1')

        # conv2
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 64, 128],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv2_1')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 128, 128],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv2_2')
        network = tl.layers.PoolLayer(network,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      pool=tf.nn.max_pool,
                                      name='vgg_pool2')

        # conv3
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 128, 256],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv3_1')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 256, 256],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv3_2')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 256, 256],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv3_3')
        network = tl.layers.PoolLayer(network,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      pool=tf.nn.max_pool,
                                      name='vgg_pool3')
        # conv4
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 256, 512],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv4_1')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 512, 512],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv4_2')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 512, 512],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv4_3')

        network = tl.layers.PoolLayer(network,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      pool=tf.nn.max_pool,
                                      name='vgg_pool4')
        conv4 = network

        # conv5
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 512, 512],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv5_1')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 512, 512],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv5_2')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 512, 512],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='vgg_conv5_3')
        network = tl.layers.PoolLayer(network,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      pool=tf.nn.max_pool,
                                      name='vgg_pool5')

        network = tl.layers.FlattenLayer(network, name='vgg_flatten')

        return conv4, network
