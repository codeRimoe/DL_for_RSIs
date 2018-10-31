# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 12:31:01 2017

This script is for training.

author: Shengjie Liu, Haowen Luo
"""

from __future__ import print_function
import SSM
import WCRN
import scipy.io
import scipy.stats
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint, History

img_file = r'PaviaU'
img_id = 'paviaU'
gt_file = r'PaviaU_gt'
gt_id = 'paviaU_gt'

samples = SSM.SSM([610, 340, 103], 5, 9, 1, -1)
samples.load(img_file, img_id, gt_file, gt_id, 0, 1)
samples.normalization()
samples.init_sam(10)

samples.save('samples.h5')       # save samples
# samples.load_file('samples.h5')  # load samples

# network: build a WCRN with 1 residual layer
model = WCRN.build(103, 9, 1, 32)

# summary and compile
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# reload model (for pre-train)
# model.load_weights('model804.h5')

num = -1                     # init loop num
total = np.zeros((100, 5))   # record OA

# training
while(1):
    if num == 50:
        break
    num += 1

    # set batch_size
    if num < 20:
        batch_size = 10
    elif num < 40:
        batch_size = 20
    else:
        batch_size = 40
    epochs = 50

#    # abandoned
#    # self_train
#    if num % 5 == 0:
#        print(len(samples.candidate['label']))
#        print('self_train!')
#        predict3 = samples.st_flush()
#        m = np.argmax(model.predict(x=predict3,verbose=0), axis=1)
#        samples.st_merge(m)

    # save model
    fpara = 'model/str_SM' + str(num) + '.h5'
    model_checkpoint = ModelCheckpoint(fpara, monitor='loss',
                                       save_best_only=True, mode='min')

    history = History()
    history = model.fit(samples.train['sample'], samples.train['label_v'],
                        batch_size=batch_size, epochs=epochs, verbose=1,
                        shuffle=True, callbacks=[model_checkpoint])

    # record OA
    total[num, 2] = min(history.history['loss'])
    total[num, 3] = max(history.history['acc'])
    predict = []
    prob = []

    # voting
    for i in range(3):
        epochs = 1
        history = model.fit(samples.train['sample'],
                            samples.train['label_v'],
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            shuffle=True)
        tmp = model.predict(x=samples.candidate['sample'], verbose=0)  # cand
        tmp2 = model.predict(x=samples.test['sample'], verbose=0)      # test
        prob.append(tmp)
        predict.append(np.argmax(tmp2, axis=1))                        # test

    predict2 = np.sum(prob, axis=0)
    maxx = scipy.stats.mode(predict)
    maxx = maxx[0].T
    maxx = maxx.reshape(maxx.shape[0])
    predict.append(maxx)
    # maxx.reshape(len(maxx),1)
    total[num, 4] = np.sum(maxx == np.argmax(samples.test['label_v'],
                           axis=1)) / len(maxx)                   # testlabel
    print(total[num, 4])                                           # print OA

    # save OA
    # np.savetxt(fpara+'.csv',np.vstack(predict).T,delimiter=',')

    model.load_weights(fpara)
    evaluation = model.evaluate(x=samples.test['sample'],
                                y=samples.test['label_v'],
                                verbose=1)              # test on test sample
    total[num, 0] = evaluation[0]
    total[num, 1] = evaluation[1]

    # Self Train
    # self_train(np.argmax(samples.test['label_v'],axis=1))

    # Active Learning
    samples.selector('SM', [10, predict2])
