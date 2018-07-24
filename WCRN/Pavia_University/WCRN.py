# -*- coding: utf-8 -*-
"""
This script is used to definr a WCRN.

author: Shengjie Liu, Haowen Luo
"""

from __future__ import print_function
import keras
from keras.models import Model
from keras.layers import Dense, Flatten, Add
from keras.layers import Conv2D, MaxPooling2D, Input, Activation
from keras.initializers import RandomNormal


# band is the number of band
# ncla is the number of classes
# nr is the number of residual layers
# nk is the number of kernels
def build(band, ncla, nr=1, nk=32):
    input_ = Input(shape=(5, 5, band))
    inputs = keras.layers.core.Permute((3, 1, 2))(input_)
    inputs = keras.layers.core.Reshape((band, 5, 5, 1))(inputs)
    x1 = keras.layers.ConvLSTM2D(filters=nk, kernel_size=(3, 3),
                                 padding='valid', activation='tanh',
                                 recurrent_activation='hard_sigmoid',
                                 use_bias=True,
                                 kernel_initializer='glorot_uniform',
                                 recurrent_initializer='orthogonal',
                                 bias_initializer='zeros')(inputs)

    x2 = keras.layers.ConvLSTM2D(filters=nk, kernel_size=(1, 1),
                                 padding='valid', activation='tanh',
                                 recurrent_activation='hard_sigmoid',
                                 use_bias=True,
                                 kernel_initializer='glorot_uniform',
                                 recurrent_initializer='orthogonal',
                                 bias_initializer='zeros')(inputs)
    x1 = keras.layers.core.Reshape((3, 3, nk))(x1)
    x2 = keras.layers.core.Reshape((5, 5, nk))(x2)
    x1 = MaxPooling2D(pool_size=(3, 3))(x1)
    x2 = MaxPooling2D(pool_size=(5, 5))(x2)
    x = keras.layers.concatenate([x1, x2], axis=3)

    while nr:
        nr -= 1
        # residual
        x1 = keras.layers.BatchNormalization(axis=-1,
                                             momentum=0.9,
                                             epsilon=0.001,
                                             center=True,
                                             scale=True,
                                             beta_initializer='zeros',
                                             gamma_initializer='ones',
                                             moving_mean_initializer='zeros',
                                             moving_variance_initializer='ones'
                                             )(x)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(2 * nk, kernel_size=(1, 1),
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x1 = keras.layers.BatchNormalization(axis=-1,
                                             momentum=0.9,
                                             epsilon=0.001,
                                             center=True,
                                             scale=True,
                                             beta_initializer='zeros',
                                             gamma_initializer='ones',
                                             moving_mean_initializer='zeros',
                                             moving_variance_initializer='ones'
                                             )(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(2 * nk, kernel_size=(1, 1),
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x1)
        x = Add()([x1, x])

#    x = Activation('relu')(x)
#    x = Dropout(0.5)(x)
#    x = Conv2D(128,kernel_size=(1,1),
#               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
#    x = Activation('relu')(x)
#    x = Dropout(0.5)(x)

    x = Flatten()(x)
    predict = Dense(ncla, activation='softmax')(x)
    model = Model(inputs=input_, outputs=predict)
    return model
