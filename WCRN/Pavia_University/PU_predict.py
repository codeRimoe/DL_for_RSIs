# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 12:31:01 2017

This script is for predicting.

author: Shengjie Liu, Haowen Luo
"""

from __future__ import print_function
import SSM
import WCRN
import numpy as np
import keras

img_file = r'PaviaU'
img_id = 'paviaU'
gt_file = r'PaviaU_gt'
gt_id = 'paviaU_gt'

samples = SSM.SSM([610, 340, 103], 5, 9, 1, -1)
samples.load(img_file, img_id, gt_file, gt_id, 0, 1)
samples.normalization()
samples.load_all_sam()

samples.save_all('all.h5')
# samples.load_all_file('all.h5')

# network
model = WCRN.build(103, 9, 1, 32)

#
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# reload
model.load_weights('model/str_SM0.h5')

# evaluation = model.evaluate(x=samples.test['sample'],
#                             y=samples.test['label_v'],
#                             verbose=1) # test on test sample

predict_map = np.argmax(model.predict(x=np.array(samples.all['sample'][:]),
                                      verbose=1), axis=1)
predict_map = predict_map.reshape(610, 340).astype(int)


# pm = np.zeros([610,340])
# j = 0
# for i in samples.all['XY']:
#     pm[i[1]][i[0]] = predict_map[j]
#     j += 1

np.savetxt('predict.asc', predict_map, fmt='%d')
tmp = ''
with open('predict.asc', 'r') as rf:
    tmp = rf.read()
with open('predict.asc', 'w') as wf:
    wf.write('ncols         340\n' +
             'nrows         610\n' +
             'xllcorner     0\n' +
             'yllcorner     0\n' +
             'cellsize      1\n' +
             'NODATA_value  -9999\n' +
             tmp)
