# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 19:01:28 2018

@author: I01-Ghorbanian
"""

import tensorflow as tf
import tensorlayer as tl
tf.reset_default_graph()


# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 10:38:54 2018

@author: I01-Ghorbanian
"""

def discriminator(input_images, is_train=True, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = tl.layers.InputLayer(input_images,
                            name='input')

        net_h0 = tl.layers.Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                        padding='SAME', W_init=w_init, name='h0/co1nv2d')

        net_h1 = tl.layers.Conv2d(net_h0, df_dim * 2, (4, 4), (2, 2), act=None,
                        padding='SAME', W_init=w_init, b_init=b_init, name='h1/con2v2d')
        
        net_h2 = tl.layers.BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h1/bat3chnorm')

        net_h3 = tl.layers.Conv2d(net_h2, df_dim * 2, (4, 4), (1, 1), act=None,
                        padding='SAME', W_init=w_init, b_init=b_init, name='h2/co14nv2d')
        
        net_h4 = tl.layers.Conv2d(net_h3, df_dim * 2, (4, 4), (1, 1), act=None,
                        padding='SAME', W_init=w_init, b_init=b_init, name='h2/c2o5nv2d')
        
        net_h5 = tl.layers.ElementwiseLayer([net_h2, net_h4], combine_fn=tf.add, name='h8/61add')
        
        net_h6 = tl.layers.BatchNormLayer(net_h5, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h2/batc7hnorm')

        net_h7 = tl.layers.Conv2d(net_h6, df_dim * 4, (4, 4), (2, 2), act=None,
                        padding='SAME', W_init=w_init, b_init=b_init, name='h3/con8v2d')
        
 
        net_h8 = tl.layers.BatchNormLayer(net_h7, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h3/bat9chnorm')

        net_h9 = tl.layers.Conv2d(net_h8, df_dim * 4, (4, 4), (1, 1), act=None,
                        padding='SAME', W_init=w_init, b_init=b_init, name='h4/conv02d')
        
        net_h10 = tl.layers.Conv2d(net_h9, df_dim * 4, (4, 4), (1, 1), act=None,
                        padding='SAME', W_init=w_init, b_init=b_init, name='h4/c3onv122d')
        
        net_h11 = tl.layers.ElementwiseLayer([net_h8, net_h10], combine_fn=tf.add, name='h812/1add')
        
        net_h12 = tl.layers.BatchNormLayer(net_h11, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h4/batc12hnorm')

        net_h13 = tl.layers.Conv2d(net_h12, df_dim * 16, (4, 4), (2, 2), act=None,
                        padding='SAME', W_init=w_init, b_init=b_init, name='h5/co23nv2d')
        
        net_h14 = tl.layers.BatchNormLayer(net_h13, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h5/batc23hnorm')

        net_h15 = tl.layers.Conv2d(net_h14, df_dim * 16, (4, 4), (1, 1), act=None,
                        padding='SAME', W_init=w_init, b_init=b_init, name='h6/con34v2d')
        
        net_h16 = tl.layers.Conv2d(net_h15, df_dim * 16, (4, 4), (1, 1), act=None,
                        padding='SAME', W_init=w_init, b_init=b_init, name='h6/con454v2d')
        
        net_h17 = tl.layers.ElementwiseLayer([net_h16, net_h14], combine_fn=tf.add, name='h8/561add')
        
        net_h18 = tl.layers.BatchNormLayer(net_h17, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h6/bat67hnorm')

        net_h19 = tl.layers.Conv2d(net_h18, df_dim * 2, (4, 4), (2, 2), act=None,
                        padding='SAME', W_init=w_init, b_init=b_init, name='h7/conv2d')

        net_h20 = tl.layers.BatchNormLayer(net_h19, is_train=is_train, gamma_init=gamma_init, name='h7/batchnorm')

        net_h21 = tl.layers.Conv2d(net_h20, df_dim * 2, (4, 4), (1, 1), act=None,
                     padding='SAME', W_init=w_init, b_init=b_init, name='h7_res/co78nv2d')
        
       # net_h15 = tl.layers.BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
       #                      is_train=is_train, gamma_init=gamma_init, name='h7_res/batchnorm')
        
        net_h22 = tl.layers.Conv2d(net_h21, df_dim * 2, (4, 4), (1, 1), act=None,
                     padding='SAME', W_init=w_init, b_init=b_init, name='h7_res/co89nv2d2')
        
        net_h23 = tl.layers.ElementwiseLayer([net_h22, net_h20], combine_fn=tf.add, name='h8/76add')
        
        net_h24 = tl.layers.Conv2d(net_h23, df_dim * 8, (4, 4), (2, 2), act=None,
                     padding='SAME', W_init=w_init, b_init=b_init, name='h7_res/co89ndfv2d2')
      #  net = tl.layers.Conv2d(net, df_dim * 8, (3, 3), (1, 1), act=None,
      #               padding='SAME', W_init=w_init, b_init=b_init, name='h7_res/conv2d3')
        
       # net = tl.layers.BatchNormLayer(net, is_train=is_train, gamma_init=gamma_init, name='h7_res/batchnorm3')

      #  net_h8 = tl.layers.ElementwiseLayer([net_h7, net], combine_fn=tf.add, name='h8/add')
        net_h24.outputs = tl.act.lrelu(net_h24.outputs, 0.2)

        net_ho = tl.layers.FlattenLayer(net_h24, name='output/flatten111')
        net_ho = tl.layers.DenseLayer(net_ho, n_units=1, act=tf.identity, W_init=w_init, name='output/dense')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)

    return net_ho, logits



def u_net_bn(x, is_train=False, reuse=False, is_refine=False):
    
    #tf.reset_default_graph() 
    
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = tl.layers.InputLayer(x, name='intyutut')
        
      #  net_h2 = tl.layers.BatchNormLayer(inputs, act=lambda x: tl.act.lrelu(x, 0.2),
      #                          is_train=is_train, gamma_init=gamma_init, name='h2tyutbtyuSDtuhnorm') 
        
      #  conv1 = tl.layers.Conv2d(inputs, n_filter=16, filter_size=(3, 3), strides=(1, 1), padding='SAME',
      #                    act=None, W_init=w_init, b_init=b_init, name='contyutuy1')
        
      #  conv2 = tl.layers.Conv2d(conv1, n_filter=16, filter_size=(3, 3), strides=(1, 1), padding='SAME',
      #                    act=None, W_init=w_init, b_init=b_init, name='conv2tutu1')
        
        conv3 = tl.layers.Conv2d(inputs, n_filter=16, filter_size=(4, 4), strides=(1, 1), padding='SAME',
                          act=None, W_init=w_init, b_init=b_init, name='conv3tyutuh1')
        ####################convolutional block
        X = conv3
        X_shortcut = X
 
       # X1 = tl.layers.Conv2d(X, n_filter=16, filter_size=(1, 1), strides=(2, 2), padding='valid',
       #                  act=None, W_init=w_init, b_init=b_init, name='ne14tae11')
    
       # X2 = tl.layers.BatchNormLayer(X1, act=lambda x: tl.act.lrelu(x, 0.2),
       #                        is_train=is_train, gamma_init=gamma_init, name='b77n122')
    
        X3 = tl.layers.Conv2d(X, n_filter=16, filter_size=(4, 4), strides=(2, 2), padding='same',
                         act=None, W_init=w_init, b_init=b_init, name='net4t4t1413')
        X4 = tl.layers.BatchNormLayer(X3, act=lambda x: tl.act.lrelu(x, 0.2),
                                  is_train=is_train, gamma_init=gamma_init, name='b657n13442')
    
        X5 = tl.layers.Conv2d(X4, n_filter=32, filter_size=(4, 4), strides=(1, 1), padding='same',
                          act=None, W_init=w_init, b_init=b_init, name='netttttttt1t1')
    
        X_shortcut1 = tl.layers.Conv2d(X_shortcut, n_filter=32, filter_size=(4, 4), strides=(2, 2), padding='same',
                                  act=None, W_init=w_init, b_init=b_init, name='nety1uty_h1')
        X_shortcut2 = tl.layers.BatchNormLayer(X_shortcut1, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='b34sda1n2')
    
        X6 = tl.layers.ElementwiseLayer([X5, X_shortcut2], combine_fn=tf.add, name='ay1tutd')
        conv1_block = tl.layers.BatchNormLayer(X6, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='br78t1n2')
        ######################
        X = conv1_block
        X_shortcut = X
 
      #  X1 = tl.layers.Conv2d(X, n_filter=32, filter_size=(1, 1), strides=(2, 2), padding='valid',
      #                   act=None, W_init=w_init, b_init=b_init, name='ne4t2ae11')
    
      #  X2 = tl.layers.BatchNormLayer(X1, act=lambda x: tl.act.lrelu(x, 0.2),
      #                         is_train=is_train, gamma_init=gamma_init, name='bn222')
    
        X3 = tl.layers.Conv2d(X, n_filter=32, filter_size=(4, 4), strides=(2, 2), padding='same',
                         act=None, W_init=w_init, b_init=b_init, name='net4t4t2413')
        X4 = tl.layers.BatchNormLayer(X3, act=lambda x: tl.act.lrelu(x, 0.2),
                                  is_train=is_train, gamma_init=gamma_init, name='bn23442')
    
        X5 = tl.layers.Conv2d(X4, n_filter=64, filter_size=(4, 4), strides=(1, 1), padding='same',
                          act=None, W_init=w_init, b_init=b_init, name='nettttttt2tt1')
    
        X_shortcut1 = tl.layers.Conv2d(X_shortcut, n_filter=64, filter_size=(4, 4), strides=(2, 2), padding='same',
                                  act=None, W_init=w_init, b_init=b_init, name='netyut2y_h1')
        X_shortcut2 = tl.layers.BatchNormLayer(X_shortcut1, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bsda22n2')
    
        X6 = tl.layers.ElementwiseLayer([X5, X_shortcut2], combine_fn=tf.add, name='ayt2utd')
        conv2_block = tl.layers.BatchNormLayer(X6, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='brtn2')
        ######################
        
        X = conv2_block
        X_shortcut = X
 
      #  X1 = tl.layers.Conv2d(X, n_filter=64, filter_size=(1, 1), strides=(2, 2), padding='valid',
      #                   act=None, W_init=w_init, b_init=b_init, name='ne4ta3e11')
    
      #  X2 = tl.layers.BatchNormLayer(X1, act=lambda x: tl.act.lrelu(x, 0.2),
      #                         is_train=is_train, gamma_init=gamma_init, name='b3n22')
    
        X3 = tl.layers.Conv2d(X, n_filter=64, filter_size=(4, 4), strides=(2, 2), padding='same',
                         act=None, W_init=w_init, b_init=b_init, name='net4t4t433')
        X4 = tl.layers.BatchNormLayer(X3, act=lambda x: tl.act.lrelu(x, 0.2),
                                  is_train=is_train, gamma_init=gamma_init, name='bn33442')
    
        X5 = tl.layers.Conv2d(X4, n_filter=128, filter_size=(4, 4), strides=(1, 1), padding='same',
                          act=None, W_init=w_init, b_init=b_init, name='nettttttt3tt1')
    
        X_shortcut1 = tl.layers.Conv2d(X_shortcut, n_filter=128, filter_size=(4, 4), strides=(2, 2), padding='same',
                                  act=None, W_init=w_init, b_init=b_init, name='netyut3y_h1')
        X_shortcut2 = tl.layers.BatchNormLayer(X_shortcut1, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bsda3n2')
    
        X6 = tl.layers.ElementwiseLayer([X5, X_shortcut2], combine_fn=tf.add, name='ay3tutd')
        conv3_block = tl.layers.BatchNormLayer(X6, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='b3rtn2')
        ######################
        
        X = conv3_block
        X_shortcut = X
 
      #  X1 = tl.layers.Conv2d(X, n_filter=128, filter_size=(1, 1), strides=(2, 2), padding='valid',
      #                   act=None, W_init=w_init, b_init=b_init, name='ne4ta4e11')
    
      #  X2 = tl.layers.BatchNormLayer(X1, act=lambda x: tl.act.lrelu(x, 0.2),
      #                         is_train=is_train, gamma_init=gamma_init, name='4bn22')
    
        X3 = tl.layers.Conv2d(X, n_filter=128, filter_size=(4, 4), strides=(2, 2), padding='same',
                         act=None, W_init=w_init, b_init=b_init, name='net4t4t4413')
        X4 = tl.layers.BatchNormLayer(X3, act=lambda x: tl.act.lrelu(x, 0.2),
                                  is_train=is_train, gamma_init=gamma_init, name='bn43442')
    
        X5 = tl.layers.Conv2d(X4, n_filter=256, filter_size=(4, 4), strides=(1, 1), padding='same',
                          act=None, W_init=w_init, b_init=b_init, name='nettttttt4tt1')
    
        X_shortcut1 = tl.layers.Conv2d(X_shortcut, n_filter=256, filter_size=(4, 4), strides=(2, 2), padding='same',
                                  act=None, W_init=w_init, b_init=b_init, name='netyuty4_h1')
        X_shortcut2 = tl.layers.BatchNormLayer(X_shortcut1, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bs4dan2')
    
        X6 = tl.layers.ElementwiseLayer([X5, X_shortcut2], combine_fn=tf.add, name='ay4tutd')
        conv4_block = tl.layers.BatchNormLayer(X6, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='br4tn2')
        ######################
        
        X = conv4_block
        X_shortcut = X
 
      #  X1 = tl.layers.Conv2d(X, n_filter=256, filter_size=(1, 1), strides=(2, 2), padding='valid',
      #                   act=None, W_init=w_init, b_init=b_init, name='ne4ta5e11')
    
      #  X2 = tl.layers.BatchNormLayer(X1, act=lambda x: tl.act.lrelu(x, 0.2),
      #                         is_train=is_train, gamma_init=gamma_init, name='5bn22')
    
        X3 = tl.layers.Conv2d(X, n_filter=256, filter_size=(4, 4), strides=(2, 2), padding='same',
                         act=None, W_init=w_init, b_init=b_init, name='net4t4t4513')
        X4 = tl.layers.BatchNormLayer(X3, act=lambda x: tl.act.lrelu(x, 0.2),
                                  is_train=is_train, gamma_init=gamma_init, name='b5n3442')
    
        X5 = tl.layers.Conv2d(X4, n_filter=512, filter_size=(4, 4), strides=(1, 1), padding='same',
                          act=None, W_init=w_init, b_init=b_init, name='nettttttt5tt1')
    
        X_shortcut1 = tl.layers.Conv2d(X_shortcut, n_filter=512, filter_size=(4, 4), strides=(2, 2), padding='same',
                                  act=None, W_init=w_init, b_init=b_init, name='netyu5ty_h1')
        X_shortcut2 = tl.layers.BatchNormLayer(X_shortcut1, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bsdan52')
    
        X6 = tl.layers.ElementwiseLayer([X5, X_shortcut2], combine_fn=tf.add, name='a5ytutd')
        conv5_block = tl.layers.BatchNormLayer(X6, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='br5n2')
        ######################
        
        X = conv5_block
        X_shortcut = X
 
      #  X1 = tl.layers.Conv2d(X, n_filter=512, filter_size=(1, 1), strides=(2, 2), padding='valid',
      #                   act=None, W_init=w_init, b_init=b_init, name='ne4t6ae11')
    
      #  X2 = tl.layers.BatchNormLayer(X1, act=lambda x: tl.act.lrelu(x, 0.2),
      #                         is_train=is_train, gamma_init=gamma_init, name='b622')
    
        X3 = tl.layers.Conv2d(X, n_filter=512, filter_size=(4, 4), strides=(2, 2), padding='same',
                         act=None, W_init=w_init, b_init=b_init, name='net4t4t4163')
        X4 = tl.layers.BatchNormLayer(X3, act=lambda x: tl.act.lrelu(x, 0.2),
                                  is_train=is_train, gamma_init=gamma_init, name='bn36442')
    
        X5 = tl.layers.Conv2d(X4, n_filter=1024, filter_size=(4, 4), strides=(1, 1), padding='same',
                          act=None, W_init=w_init, b_init=b_init, name='nettttttt6tt1')
    
        X_shortcut1 = tl.layers.Conv2d(X_shortcut, n_filter=1024, filter_size=(4, 4), strides=(2, 2), padding='same',
                                  act=None, W_init=w_init, b_init=b_init, name='netyu6ty_h1')
        X_shortcut2 = tl.layers.BatchNormLayer(X_shortcut1, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bsdan62')
    
        X6 = tl.layers.ElementwiseLayer([X5, X_shortcut2], combine_fn=tf.add, name='ay6tutd')
        conv6_block = tl.layers.BatchNormLayer(X6, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='b6rtn2')
        ######################
        
        
        conv4 = tl.layers.Conv2d(conv6_block, n_filter=2048, filter_size=(4, 4), strides=(1, 1), padding='same',
                          act=None, W_init=w_init, b_init=b_init, name='conv1utyuth1')
        
     #   conv4 = tl.layers.Conv2d(conv4, n_filter=2048, filter_size=(4, 4), strides=(1, 1), padding='same',
     #                     act=None, W_init=w_init, b_init=b_init, name='c2onv1utyuth1')
        
        ################ENCODER
        X = conv4
        X_shortcut = X

      #  X1 = tl.layers.DeConv2d(X, n_filter=1024, filter_size=(3, 3), out_size=(8, 8), strides=(2, 2), padding='SAME',
      #                 act=None, W_init=w_init, b_init=b_init, name='d4t41tnv6')
                       
      #  X2 = tl.layers.BatchNormLayer(X1, act=lambda x: tl.act.lrelu(x, 0.2),
      #                         is_train=is_train, gamma_init=gamma_init, name='1by55')

        X3 = tl.layers.Conv2d(X, n_filter=1024, filter_size=(4, 4), strides=(1, 1), padding='same',
                         act=None, W_init=w_init, b_init=b_init, name='nat4t1h1')
        X4 = tl.layers.BatchNormLayer(X3, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='byy21345')
    
        X5 = tl.layers.DeConv2d(X4, n_filter=1024, filter_size=(4, 4), out_size=(8, 8), strides=(2, 2), padding='same',
                          act=None, W_init=w_init, b_init=b_init, name='na4t4t1h1')
    
        X6 = tl.layers.BatchNormLayer(X5, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='byy177y')
   
        X_shortcut1 = tl.layers.DeConv2d(X_shortcut, n_filter=1024, filter_size=(4, 4), out_size=(8, 8), strides=(2, 2), padding='SAME',
                                  act=None, W_init=w_init, b_init=b_init, name='nea14ta4t1')
        X_shortcut2 = tl.layers.BatchNormLayer(X_shortcut1, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='brr1r7y')
    
        X7 = tl.layers.ElementwiseLayer([X6, X_shortcut2], combine_fn=tf.add, name='ee1eeedd')
        encoder_block1 = tl.layers.BatchNormLayer(X7, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='b7641577y')
        ######################
        Concat1 = tl.layers.ConcatLayer([encoder_block1, conv5_block], concat_dim=3, name='conctutut1')
        
        
        ################ENCODER
        X = Concat1
        X_shortcut = X

      #  X1 = tl.layers.DeConv2d(X, n_filter=512, filter_size=(3, 3), out_size=(16, 16), strides=(2, 2), padding='SAME',
      #                 act=None, W_init=w_init, b_init=b_init, name='d4t42tnv6')
                       
      #  X2 = tl.layers.BatchNormLayer(X1, act=lambda x: tl.act.lrelu(x, 0.2),
      #                         is_train=is_train, gamma_init=gamma_init, name='by255')

        X3 = tl.layers.Conv2d(X, n_filter=512, filter_size=(4, 4), strides=(1, 1), padding='same',
                         act=None, W_init=w_init, b_init=b_init, name='nat4th21')
        X4 = tl.layers.BatchNormLayer(X3, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='byy22345')
    
        X5 = tl.layers.DeConv2d(X4, n_filter=512, filter_size=(4, 4), out_size=(16, 16), strides=(2, 2), padding='same',
                          act=None, W_init=w_init, b_init=b_init, name='na4t24th1')
    
        X6 = tl.layers.BatchNormLayer(X5, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='byy277y')
   
        X_shortcut1 = tl.layers.DeConv2d(X_shortcut, n_filter=512, filter_size=(4, 4), out_size=(16, 16), strides=(2, 2), padding='SAME',
                                  act=None, W_init=w_init, b_init=b_init, name='nea42ta4t1')
        X_shortcut2 = tl.layers.BatchNormLayer(X_shortcut1, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='br2rr7y')
    
        X7 = tl.layers.ElementwiseLayer([X6, X_shortcut2], combine_fn=tf.add, name='ee2eeedd')
        encoder_block2 = tl.layers.BatchNormLayer(X7, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='b7624577y')
        ##################
        Concat2 = tl.layers.ConcatLayer([encoder_block2, conv4_block], concat_dim=3, name='contyu2tut2')
        
                
        ################ENCODER
        X = Concat2
        X_shortcut = X

      #  X1 = tl.layers.DeConv2d(X, n_filter=256, filter_size=(3, 3), out_size=(32, 32), strides=(2, 2), padding='SAME',
      #                 act=None, W_init=w_init, b_init=b_init, name='d4t4t3nv6')
                       
      #  X2 = tl.layers.BatchNormLayer(X1, act=lambda x: tl.act.lrelu(x, 0.2),
      #                         is_train=is_train, gamma_init=gamma_init, name='by535')

        X3 = tl.layers.Conv2d(X, n_filter=256, filter_size=(4, 4), strides=(1, 1), padding='same',
                         act=None, W_init=w_init, b_init=b_init, name='nat34th1')
        X4 = tl.layers.BatchNormLayer(X3, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='byy23345')
    
        X5 = tl.layers.DeConv2d(X4, n_filter=256, filter_size=(4, 4), out_size=(32, 32), strides=(2, 2), padding='same',
                          act=None, W_init=w_init, b_init=b_init, name='na4t4t3h1')
    
        X6 = tl.layers.BatchNormLayer(X5, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='by3y77y')
   
        X_shortcut1 = tl.layers.DeConv2d(X_shortcut, n_filter=256, filter_size=(4, 4), out_size=(32, 32), strides=(2, 2), padding='SAME',
                                  act=None, W_init=w_init, b_init=b_init, name='nea34ta4t1')
        X_shortcut2 = tl.layers.BatchNormLayer(X_shortcut1, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='brrr73y')
    
        X7 = tl.layers.ElementwiseLayer([X6, X_shortcut2], combine_fn=tf.add, name='3eeeeedd')
        encoder_block3 = tl.layers.BatchNormLayer(X7, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='b76435377y')
        ##################
        Concat3 = tl.layers.ConcatLayer([encoder_block3, conv3_block], concat_dim=3, name='tutuoncat2')
        
                
        ################ENCODER
        X = Concat3
        X_shortcut = X

      #  X1 = tl.layers.DeConv2d(X, n_filter=128, filter_size=(3, 3), out_size=(64, 64), strides=(2, 2), padding='SAME',
      #                 act=None, W_init=w_init, b_init=b_init, name='d4t44tnv6')
                       
      #  X2 = tl.layers.BatchNormLayer(X1, act=lambda x: tl.act.lrelu(x, 0.2),
      #                         is_train=is_train, gamma_init=gamma_init, name='b4y55')

        X3 = tl.layers.Conv2d(X, n_filter=128, filter_size=(4, 4), strides=(1, 1), padding='same',
                         act=None, W_init=w_init, b_init=b_init, name='nat4t4h1')
        X4 = tl.layers.BatchNormLayer(X3, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='by4y2345')
    
        X5 = tl.layers.DeConv2d(X4, n_filter=128, filter_size=(4, 4), out_size=(64, 64), strides=(2, 2), padding='same',
                          act=None, W_init=w_init, b_init=b_init, name='na4t4t4h1')
    
        X6 = tl.layers.BatchNormLayer(X5, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='byy477y')
   
        X_shortcut1 = tl.layers.DeConv2d(X_shortcut, n_filter=128, filter_size=(4, 4), out_size=(64, 64), strides=(2, 2), padding='SAME',
                                  act=None, W_init=w_init, b_init=b_init, name='ne4a4ta4t1')
        X_shortcut2 = tl.layers.BatchNormLayer(X_shortcut1, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='brr4r7y')
    
        X7 = tl.layers.ElementwiseLayer([X6, X_shortcut2], combine_fn=tf.add, name='ee4eeedd')
        encoder_block4 = tl.layers.BatchNormLayer(X7, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='b7644577y')
        ##################
        Concat4 = tl.layers.ConcatLayer([encoder_block4, conv2_block], concat_dim=3, name='conctutut')
        
                
        ################ENCODER
        X = Concat4
        X_shortcut = X

     #   X1 = tl.layers.DeConv2d(X, n_filter=64, filter_size=(3, 3), out_size=(128, 128), strides=(2, 2), padding='SAME',
     #                  act=None, W_init=w_init, b_init=b_init, name='d4t4t5nv6')
                       
     #   X2 = tl.layers.BatchNormLayer(X1, act=lambda x: tl.act.lrelu(x, 0.2),
     #                          is_train=is_train, gamma_init=gamma_init, name='b55y55')

        X3 = tl.layers.Conv2d(X, n_filter=64, filter_size=(4, 4), strides=(1, 1), padding='same',
                         act=None, W_init=w_init, b_init=b_init, name='nat45th1')
        X4 = tl.layers.BatchNormLayer(X3, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='byy25345')
    
        X5 = tl.layers.DeConv2d(X4, n_filter=64, filter_size=(4, 4), out_size=(128, 128), strides=(2, 2), padding='same',
                          act=None, W_init=w_init, b_init=b_init, name='na4t45th1')
    
        X6 = tl.layers.BatchNormLayer(X5, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='byy757y')
   
        X_shortcut1 = tl.layers.DeConv2d(X_shortcut, n_filter=64, filter_size=(4, 4), out_size=(128, 128), strides=(2, 2), padding='SAME',
                                  act=None, W_init=w_init, b_init=b_init, name='nea45ta4t1')
        X_shortcut2 = tl.layers.BatchNormLayer(X_shortcut1, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='brrr57y')
    
        X7 = tl.layers.ElementwiseLayer([X6, X_shortcut2], combine_fn=tf.add, name='ee5eeedd')
        encoder_block5 = tl.layers.BatchNormLayer(X7, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='b7645577y')
        ##################
        Concat5 = tl.layers.ConcatLayer([encoder_block5, conv1_block], concat_dim=3, name='ctyutut')
        
                
        ################ENCODER
        X = Concat5
        X_shortcut = X

     #   X1 = tl.layers.DeConv2d(X, n_filter=32, filter_size=(3, 3), out_size=(256, 256), strides=(2, 2), padding='SAME',
     #                  act=None, W_init=w_init, b_init=b_init, name='d4t46tnv6')
                       
     #   X2 = tl.layers.BatchNormLayer(X1, act=lambda x: tl.act.lrelu(x, 0.2),
     #                          is_train=is_train, gamma_init=gamma_init, name='b6y55')

        X3 = tl.layers.Conv2d(X, n_filter=32, filter_size=(4, 4), strides=(1, 1), padding='same',
                         act=None, W_init=w_init, b_init=b_init, name='nat4t6h1')
        X4 = tl.layers.BatchNormLayer(X3, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='by6y2345')
    
        X5 = tl.layers.DeConv2d(X4, n_filter=32, filter_size=(4, 4), out_size=(256, 256), strides=(2, 2), padding='same',
                          act=None, W_init=w_init, b_init=b_init, name='na4t4t6h1')
    
        X6 = tl.layers.BatchNormLayer(X5, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='by6y77y')
   
        X_shortcut1 = tl.layers.DeConv2d(X_shortcut, n_filter=32, filter_size=(4, 4), out_size=(256, 256), strides=(2, 2), padding='SAME',
                                  act=None, W_init=w_init, b_init=b_init, name='nea64ta4t1')
        X_shortcut2 = tl.layers.BatchNormLayer(X_shortcut1, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='brrr67y')
    
        X7 = tl.layers.ElementwiseLayer([X6, X_shortcut2], combine_fn=tf.add, name='e6eeeedd')
        encoder_block6 = tl.layers.BatchNormLayer(X7, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='b7646577y')
        ##################
        
        
        conv5 = tl.layers.Conv2d(encoder_block6, n_filter=64, filter_size=(4, 4), strides=(1, 1), padding='SAME',
                          act=None, W_init=w_init, b_init=b_init, name='cbcvb_h1')
        
       # conv6 = tl.layers.Conv2d(conv5, n_filter=16, filter_size=(4, 4), strides=(1, 1), padding='SAME',
       #                   act=None, W_init=w_init, b_init=b_init, name='cocvbc')

        if is_refine:
            out = tl.layers.Conv2d(conv5, n_filter=1, filter_size=(4, 4), strides=(1, 1), padding='SAME', act=tf.nn.tanh, name='out')
            out = tl.layers.ElementwiseLayer([out, inputs], combine_fn = tf.add, name = 'add_for_refine')
            out.outputs = tl.act.ramp(out.outputs, v_min=-1, v_max=1)
        else :
            out = tl.layers.Conv2d(conv5, n_filter=1, filter_size=(1, 1), strides=(1, 1), padding='SAME', act=tf.nn.tanh, name='out')
    return out


def vgg16_cnn_emb(t_image, reuse=False):
    
    #tf.reset_default_graph() 
    
    with tf.variable_scope("vgg16_cnn", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        t_image = (t_image + 1) * 127.5  # convert input of [-1, 1] to [0, 255]

        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
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


if __name__ == "__main__":
    pass
        
        