"""
methods.py is written to define Deep models
"""

import tensorflow as tf
import tensorlayer as tl
tf.reset_default_graph()

class DeepResidualModel:
    """
    DeepResidualModel is to define model
    """
    def __init__(self):
        """
        The get_args method is written to return config arguments
        """
        self.gamma_init = tf.random_normal_initializer(1., 0.02)
        self.b_init_dis = None
        self.b_init_gen = tf.constant_initializer(value=0.0)
        self.w_init_dis = tf.random_normal_initializer(stddev=0.02)
        self.w_init_gen = tf.truncated_normal_initializer(stddev=0.01)
        self.df_dim = 64

    def discriminator(self, input_images, is_train=True, reuse=False):
        """
        The vgg16_cnn_emb method is written to define Discriminative model
        :param input_images: 4D tensor
        :param is_train: bool
        :param reuse: bool
        :return logits: int
        :return net_ho: 1D tensor
        """

        with tf.variable_scope("discriminator", reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            net_in = tl.layers.InputLayer(input_images, name='input')

            net_h0 = tl.layers.Conv2d(
                net_in, self.df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=self.w_init_dis, name='h0/co1nv2d'
            )

            net_h1 = tl.layers.Conv2d(
                net_h0, self.df_dim * 2, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=self.w_init_dis, b_init=self.b_init_dis, name='h1/con2v2d'
            )

            net_h2 = tl.layers.BatchNormLayer(
                net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='h1/bat3chnorm'
            )

            net_h3 = tl.layers.Conv2d(
                net_h2, self.df_dim * 2, (4, 4), (1, 1), act=None,
                padding='SAME', W_init=self.w_init_dis, b_init=self.b_init_dis, name='h2/co14nv2d'
            )

            net_h4 = tl.layers.Conv2d(
                net_h3, self.df_dim * 2, (4, 4), (1, 1), act=None,
                padding='SAME', W_init=self.w_init_dis, b_init=self.b_init_dis, name='h2/c2o5nv2d'
            )

            net_h5 = tl.layers.ElementwiseLayer(
                [net_h2, net_h4], combine_fn=tf.add, name='h8/61add'
            )

            net_h6 = tl.layers.BatchNormLayer(
                net_h5, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='h2/batc7hnorm'
            )

            net_h7 = tl.layers.Conv2d(
                net_h6, self.df_dim * 4, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=self.w_init_dis, b_init=self.b_init_dis, name='h3/con8v2d'
            )


            net_h8 = tl.layers.BatchNormLayer(
                net_h7, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='h3/bat9chnorm'
            )

            net_h9 = tl.layers.Conv2d(
                net_h8, self.df_dim * 4, (4, 4), (1, 1), act=None,
                padding='SAME', W_init=self.w_init_dis, b_init=self.b_init_dis, name='h4/conv02d'
            )

            net_h10 = tl.layers.Conv2d(
                net_h9, self.df_dim * 4, (4, 4), (1, 1), act=None,
                padding='SAME', W_init=self.w_init_dis, b_init=self.b_init_dis, name='h4/c3onv122d'
            )

            net_h11 = tl.layers.ElementwiseLayer(
                [net_h8, net_h10], combine_fn=tf.add, name='h812/1add'
            )

            net_h12 = tl.layers.BatchNormLayer(
                net_h11, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='h4/batc12hnorm'
            )

            net_h13 = tl.layers.Conv2d(
                net_h12, self.df_dim * 16, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=self.w_init_dis, b_init=self.b_init_dis, name='h5/co23nv2d'
            )

            net_h14 = tl.layers.BatchNormLayer(
                net_h13, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='h5/batc23hnorm'
            )

            net_h15 = tl.layers.Conv2d(
                net_h14, self.df_dim * 16, (4, 4), (1, 1), act=None,
                padding='SAME', W_init=self.w_init_dis, b_init=self.b_init_dis, name='h6/con34v2d'
            )

            net_h16 = tl.layers.Conv2d(
                net_h15, self.df_dim * 16, (4, 4), (1, 1), act=None,
                padding='SAME', W_init=self.w_init_dis, b_init=self.b_init_dis, name='h6/con454v2d'
            )

            net_h17 = tl.layers.ElementwiseLayer(
                [net_h16, net_h14], combine_fn=tf.add, name='h8/561add'
            )

            net_h18 = tl.layers.BatchNormLayer(
                net_h17, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='h6/bat67hnorm'
            )

            net_h19 = tl.layers.Conv2d(
                net_h18, self.df_dim * 2, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=self.w_init_dis, b_init=self.b_init_dis, name='h7/conv2d'
            )

            net_h20 = tl.layers.BatchNormLayer(
                net_h19, is_train=is_train, gamma_init=self.gamma_init, name='h7/batchnorm'
            )

            net_h21 = tl.layers.Conv2d(
                net_h20, self.df_dim * 2, (4, 4), (1, 1), act=None,
                padding='SAME', W_init=self.w_init_dis,
                b_init=self.b_init_dis, name='h7_res/co78nv2d'
            )

            net_h22 = tl.layers.Conv2d(
                net_h21, self.df_dim * 2, (4, 4), (1, 1), act=None,
                padding='SAME', W_init=self.w_init_dis,
                b_init=self.b_init_dis, name='h7_res/co89nv2d2'
            )

            net_h23 = tl.layers.ElementwiseLayer(
                [net_h22, net_h20], combine_fn=tf.add, name='h8/76add'
            )

            net_h24 = tl.layers.Conv2d(
                net_h23, self.df_dim * 8, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=self.w_init_dis,
                b_init=self.b_init_dis, name='h7_res/co89ndfv2d2'
            )

            net_h24.outputs = tl.act.lrelu(net_h24.outputs, 0.2)

            net_ho = tl.layers.FlattenLayer(net_h24, name='output/flatten111')
            net_ho = tl.layers.DenseLayer(
                net_ho, n_units=1, act=tf.identity, W_init=self.w_init_dis, name='output/dense'
            )
            logits = net_ho.outputs
            net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)

        return net_ho, logits

    def u_net_bn(self, input_unet, is_train=False, reuse=False, is_refine=False):
        """
        The u_net_bn method is written to define Generative model
        :param input_unet: 4D tensor
        :param is_train: bool
        :param reuse: bool
        :param is_refine: bool
        :return out: 4D tensor
        """
        #tf.reset_default_graph()
        with tf.variable_scope("u_net", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            inputs = tl.layers.InputLayer(input_unet, name='intyutut')

            conv3 = tl.layers.Conv2d(
                inputs, n_filter=16, filter_size=(4, 4), strides=(1, 1), padding='SAME',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='conv3tyutuh1'
            )
            ############## convolutional block #################
            out = conv3
            out_shortcut = out

            out3 = tl.layers.Conv2d(
                out, n_filter=16, filter_size=(4, 4), strides=(2, 2), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='net4t4t1413'
            )
            out4 = tl.layers.BatchNormLayer(
                out3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='b657n13442'
            )

            out5 = tl.layers.Conv2d(
                out4, n_filter=32, filter_size=(4, 4), strides=(1, 1), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='netttttttt1t1'
            )

            out_shortcut1 = tl.layers.Conv2d(
                out_shortcut, n_filter=32, filter_size=(4, 4), strides=(2, 2), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='nety1uty_h1'
            )
            out_shortcut2 = tl.layers.BatchNormLayer(
                out_shortcut1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='b34sda1n2'
            )

            out6 = tl.layers.ElementwiseLayer(
                [out5, out_shortcut2], combine_fn=tf.add, name='ay1tutd'
            )
            conv1_block = tl.layers.BatchNormLayer(
                out6, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='br78t1n2'
            )

            out = conv1_block
            out_shortcut = out

            out3 = tl.layers.Conv2d(
                out, n_filter=32, filter_size=(4, 4), strides=(2, 2), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='net4t4t2413'
            )
            out4 = tl.layers.BatchNormLayer(
                out3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='bn23442'
            )

            out5 = tl.layers.Conv2d(
                out4, n_filter=64, filter_size=(4, 4), strides=(1, 1), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='nettttttt2tt1'
            )

            out_shortcut1 = tl.layers.Conv2d(
                out_shortcut, n_filter=64, filter_size=(4, 4), strides=(2, 2), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='netyut2y_h1'
            )
            out_shortcut2 = tl.layers.BatchNormLayer(
                out_shortcut1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='bsda22n2'
            )

            out6 = tl.layers.ElementwiseLayer(
                [out5, out_shortcut2], combine_fn=tf.add, name='ayt2utd'
            )
            conv2_block = tl.layers.BatchNormLayer(
                out6, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='brtn2'
            )
            ######################

            out = conv2_block
            out_shortcut = out

            out3 = tl.layers.Conv2d(
                out, n_filter=64, filter_size=(4, 4), strides=(2, 2), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='net4t4t433'
            )
            out4 = tl.layers.BatchNormLayer(
                out3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='bn33442'
            )

            out5 = tl.layers.Conv2d(
                out4, n_filter=128, filter_size=(4, 4), strides=(1, 1), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='nettttttt3tt1'
            )

            out_shortcut1 = tl.layers.Conv2d(
                out_shortcut, n_filter=128, filter_size=(4, 4), strides=(2, 2), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='netyut3y_h1'
            )
            out_shortcut2 = tl.layers.BatchNormLayer(
                out_shortcut1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='bsda3n2'
            )

            out6 = tl.layers.ElementwiseLayer(
                [out5, out_shortcut2], combine_fn=tf.add, name='ay3tutd'
            )
            conv3_block = tl.layers.BatchNormLayer(
                out6, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='b3rtn2'
            )
            ######################

            out = conv3_block
            out_shortcut = out

            out3 = tl.layers.Conv2d(
                out, n_filter=128, filter_size=(4, 4), strides=(2, 2), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='net4t4t4413'
            )
            out4 = tl.layers.BatchNormLayer(
                out3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='bn43442'
            )

            out5 = tl.layers.Conv2d(
                out4, n_filter=256, filter_size=(4, 4), strides=(1, 1), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='nettttttt4tt1'
            )

            out_shortcut1 = tl.layers.Conv2d(
                out_shortcut, n_filter=256, filter_size=(4, 4), strides=(2, 2), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='netyuty4_h1'
            )
            out_shortcut2 = tl.layers.BatchNormLayer(
                out_shortcut1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='bs4dan2'
            )

            out6 = tl.layers.ElementwiseLayer(
                [out5, out_shortcut2], combine_fn=tf.add, name='ay4tutd'
            )
            conv4_block = tl.layers.BatchNormLayer(
                out6, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='br4tn2'
            )
            ######################

            out = conv4_block
            out_shortcut = out

            out3 = tl.layers.Conv2d(
                out, n_filter=256, filter_size=(4, 4), strides=(2, 2), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='net4t4t4513'
            )
            out4 = tl.layers.BatchNormLayer(
                out3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='b5n3442'
            )

            out5 = tl.layers.Conv2d(
                out4, n_filter=512, filter_size=(4, 4), strides=(1, 1), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='nettttttt5tt1'
            )

            out_shortcut1 = tl.layers.Conv2d(
                out_shortcut, n_filter=512, filter_size=(4, 4), strides=(2, 2), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='netyu5ty_h1'
            )
            out_shortcut2 = tl.layers.BatchNormLayer(
                out_shortcut1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='bsdan52'
            )

            out6 = tl.layers.ElementwiseLayer(
                [out5, out_shortcut2], combine_fn=tf.add, name='a5ytutd'
            )
            conv5_block = tl.layers.BatchNormLayer(
                out6, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='br5n2'
            )
            ######################

            out = conv5_block
            out_shortcut = out

            out3 = tl.layers.Conv2d(
                out, n_filter=512, filter_size=(4, 4), strides=(2, 2), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='net4t4t4163'
            )
            out4 = tl.layers.BatchNormLayer(
                out3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='bn36442'
            )

            out5 = tl.layers.Conv2d(
                out4, n_filter=1024, filter_size=(4, 4), strides=(1, 1), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='nettttttt6tt1'
            )

            out_shortcut1 = tl.layers.Conv2d(
                out_shortcut, n_filter=1024, filter_size=(4, 4), strides=(2, 2), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='netyu6ty_h1'
            )
            out_shortcut2 = tl.layers.BatchNormLayer(
                out_shortcut1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='bsdan62'
            )

            out6 = tl.layers.ElementwiseLayer(
                [out5, out_shortcut2], combine_fn=tf.add, name='ay6tutd'
            )
            conv6_block = tl.layers.BatchNormLayer(
                out6, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='b6rtn2'
            )

            conv4 = tl.layers.Conv2d(
                conv6_block, n_filter=2048, filter_size=(4, 4), strides=(1, 1), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='conv1utyuth1'
            )

            ################ ENCODER ###############
            out = conv4
            out_shortcut = out

            out3 = tl.layers.Conv2d(
                out, n_filter=1024, filter_size=(4, 4), strides=(1, 1), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='nat4t1h1'
            )
            out4 = tl.layers.BatchNormLayer(
                out3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='byy21345'
            )

            out5 = tl.layers.DeConv2d(
                out4, n_filter=1024, filter_size=(4, 4), out_size=(8, 8),
                strides=(2, 2), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='na4t4t1h1'
            )

            out6 = tl.layers.BatchNormLayer(
                out5, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='byy177y'
            )

            out_shortcut1 = tl.layers.DeConv2d(
                out_shortcut, n_filter=1024, filter_size=(4, 4), out_size=(8, 8),
                strides=(2, 2), padding='SAME',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='nea14ta4t1'
            )
            out_shortcut2 = tl.layers.BatchNormLayer(
                out_shortcut1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='brr1r7y'
            )

            out7 = tl.layers.ElementwiseLayer(
                [out6, out_shortcut2], combine_fn=tf.add, name='ee1eeedd'
            )
            encoder_block1 = tl.layers.BatchNormLayer(
                out7, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='b7641577y'
            )
            ######################
            concat1 = tl.layers.ConcatLayer(
                [encoder_block1, conv5_block], concat_dim=3, name='conctutut1'
            )

            out = concat1
            out_shortcut = out

            out3 = tl.layers.Conv2d(
                out, n_filter=512, filter_size=(4, 4), strides=(1, 1), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='nat4th21'
            )
            out4 = tl.layers.BatchNormLayer(
                out3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='byy22345'
            )

            out5 = tl.layers.DeConv2d(
                out4, n_filter=512, filter_size=(4, 4), out_size=(16, 16),
                strides=(2, 2), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='na4t24th1'
            )

            out6 = tl.layers.BatchNormLayer(
                out5, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='byy277y'
            )

            out_shortcut1 = tl.layers.DeConv2d(
                out_shortcut, n_filter=512, filter_size=(4, 4),
                out_size=(16, 16), strides=(2, 2), padding='SAME',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='nea42ta4t1'
            )
            out_shortcut2 = tl.layers.BatchNormLayer(
                out_shortcut1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='br2rr7y'
            )

            out7 = tl.layers.ElementwiseLayer(
                [out6, out_shortcut2], combine_fn=tf.add, name='ee2eeedd'
            )
            encoder_block2 = tl.layers.BatchNormLayer(
                out7, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='b7624577y'
            )

            concat2 = tl.layers.ConcatLayer(
                [encoder_block2, conv4_block], concat_dim=3, name='contyu2tut2'
            )

            out = concat2
            out_shortcut = out

            out3 = tl.layers.Conv2d(
                out, n_filter=256, filter_size=(4, 4), strides=(1, 1), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='nat34th1'
            )
            out4 = tl.layers.BatchNormLayer(
                out3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='byy23345'
            )

            out5 = tl.layers.DeConv2d(
                out4, n_filter=256, filter_size=(4, 4), out_size=(32, 32), strides=(2, 2),
                padding='same', act=None, W_init=self.w_init_gen,
                b_init=self.b_init_gen, name='na4t4t3h1'
            )

            out6 = tl.layers.BatchNormLayer(
                out5, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='by3y77y'
            )

            out_shortcut1 = tl.layers.DeConv2d(
                out_shortcut, n_filter=256, filter_size=(4, 4), out_size=(32, 32),
                strides=(2, 2), padding='SAME', act=None, W_init=self.w_init_gen,
                b_init=self.b_init_gen, name='nea34ta4t1'
            )
            out_shortcut2 = tl.layers.BatchNormLayer(
                out_shortcut1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='brrr73y'
            )

            out7 = tl.layers.ElementwiseLayer(
                [out6, out_shortcut2], combine_fn=tf.add, name='3eeeeedd'
            )
            encoder_block3 = tl.layers.BatchNormLayer(
                out7, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='b76435377y'
            )

            concat3 = tl.layers.ConcatLayer(
                [encoder_block3, conv3_block], concat_dim=3, name='tutuoncat2'
            )

            out = concat3
            out_shortcut = out

            out3 = tl.layers.Conv2d(
                out, n_filter=128, filter_size=(4, 4), strides=(1, 1), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='nat4t4h1'
            )
            out4 = tl.layers.BatchNormLayer(
                out3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='by4y2345'
            )

            out5 = tl.layers.DeConv2d(
                out4, n_filter=128, filter_size=(4, 4), out_size=(64, 64), strides=(2, 2),
                padding='same', act=None, W_init=self.w_init_gen,
                b_init=self.b_init_gen, name='na4t4t4h1'
            )

            out6 = tl.layers.BatchNormLayer(
                out5, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='byy477y'
            )

            out_shortcut1 = tl.layers.DeConv2d(
                out_shortcut, n_filter=128, filter_size=(4, 4), out_size=(64, 64),
                strides=(2, 2), padding='SAME', act=None, W_init=self.w_init_gen,
                b_init=self.b_init_gen, name='ne4a4ta4t1'
            )
            out_shortcut2 = tl.layers.BatchNormLayer(
                out_shortcut1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='brr4r7y'
            )

            out7 = tl.layers.ElementwiseLayer(
                [out6, out_shortcut2], combine_fn=tf.add, name='ee4eeedd'
            )
            encoder_block4 = tl.layers.BatchNormLayer(
                out7, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='b7644577y'
            )

            concat4 = tl.layers.ConcatLayer(
                [encoder_block4, conv2_block], concat_dim=3, name='conctutut'
            )

            out = concat4
            out_shortcut = out

            out3 = tl.layers.Conv2d(
                out, n_filter=64, filter_size=(4, 4), strides=(1, 1), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='nat45th1'
            )
            out4 = tl.layers.BatchNormLayer(
                out3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='byy25345'
            )

            out5 = tl.layers.DeConv2d(
                out4, n_filter=64, filter_size=(4, 4), out_size=(128, 128), strides=(2, 2),
                padding='same', act=None, W_init=self.w_init_gen,
                b_init=self.b_init_gen, name='na4t45th1'
            )

            out6 = tl.layers.BatchNormLayer(
                out5, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='byy757y'
            )

            out_shortcut1 = tl.layers.DeConv2d(
                out_shortcut, n_filter=64, filter_size=(4, 4), out_size=(128, 128),
                strides=(2, 2), padding='SAME', act=None, W_init=self.w_init_gen,
                b_init=self.b_init_gen, name='nea45ta4t1'
            )
            out_shortcut2 = tl.layers.BatchNormLayer(
                out_shortcut1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='brrr57y'
            )

            out7 = tl.layers.ElementwiseLayer(
                [out6, out_shortcut2], combine_fn=tf.add, name='ee5eeedd'
            )
            encoder_block5 = tl.layers.BatchNormLayer(
                out7, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='b7645577y'
            )
            ##################
            concat5 = tl.layers.ConcatLayer(
                [encoder_block5, conv1_block], concat_dim=3, name='ctyutut'
            )

            out = concat5
            out_shortcut = out

            out3 = tl.layers.Conv2d(
                out, n_filter=32, filter_size=(4, 4), strides=(1, 1), padding='same',
                act=None, W_init=self.w_init_gen, b_init=self.b_init_gen, name='nat4t6h1'
            )
            out4 = tl.layers.BatchNormLayer(
                out3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='by6y2345'
            )

            out5 = tl.layers.DeConv2d(
                out4, n_filter=32, filter_size=(4, 4), out_size=(256, 256),
                strides=(2, 2), padding='same', act=None, W_init=self.w_init_gen,
                b_init=self.b_init_gen, name='na4t4t6h1'
            )

            out6 = tl.layers.BatchNormLayer(
                out5, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='by6y77y'
            )

            out_shortcut1 = tl.layers.DeConv2d(
                out_shortcut, n_filter=32, filter_size=(4, 4), out_size=(256, 256),
                strides=(2, 2), padding='SAME', act=None, W_init=self.w_init_gen,
                b_init=self.b_init_gen, name='nea64ta4t1'
            )
            out_shortcut2 = tl.layers.BatchNormLayer(
                out_shortcut1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='brrr67y'
            )

            out7 = tl.layers.ElementwiseLayer(
                [out6, out_shortcut2], combine_fn=tf.add, name='e6eeeedd'
            )
            encoder_block6 = tl.layers.BatchNormLayer(
                out7, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=self.gamma_init, name='b7646577y'
            )

            conv5 = tl.layers.Conv2d(
                encoder_block6, n_filter=64, filter_size=(4, 4), strides=(1, 1),
                padding='SAME', act=None, W_init=self.w_init_gen,
                b_init=self.b_init_gen, name='cbcvb_h1'
            )

            if is_refine:
                out = tl.layers.Conv2d(
                    conv5, n_filter=1, filter_size=(4, 4), strides=(1, 1),
                    padding='SAME', act=tf.nn.tanh, name='out'
                )
                out = tl.layers.ElementwiseLayer(
                    [out, inputs], combine_fn = tf.add, name = 'add_for_refine'
                )
                out.outputs = tl.act.ramp(
                    out.outputs, v_min=-1, v_max=1
                )
            else :
                out = tl.layers.Conv2d(
                    conv5, n_filter=1, filter_size=(1, 1), strides=(1, 1),
                    padding='SAME', act=tf.nn.tanh, name='out'
                )
        return out
        