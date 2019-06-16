# -*- coding: utf-8 -*-

# Sample-set Maker(SSM) V5.0
# Author: Yue H.W. Luo
# Mail: yue.rimoe@gmail.com
# License : http://www.apache.org/licenses/LICENSE-2.0
# Reference:

'''
This is a class definition scipts for Sample-set Maker. SSM is a sample maker
for RSI (Remote Sencing Image) classification, specifically for deep learning
classification algorithm.
With SSM, you can easily load a RSI and get samples for training. Every Sample
made by SSM is a N*N sub-image, which can be a sample of its center pixel for
CNN/ResNet classifier.
Also, AL (Active Learning) is supported in SSM.
'''

# Usage:
#  1.import model:
#           > from SSM import *
#  2.initial SSM:
#      SSM(IMGSize, SAMN, CLAN[, AUG=1, RSEED=-1]):
#        ` IMGSize: the ROW, COL, BAND of image: [ROW, COL, BAND]
#        ` SAMN: the size of sample: NS
#        ` CLAM: the number of classes: NC
#        ` AUG: Data Augmentation, 1 for augmented, 0 for no augmented
#        ` RSEED: random seed, int required, [default: -1 (time seed)]
#            > samples = SSM([340, 610, 103], 5, 9, 1, -1)
#  3.load image:
#      samples.load(img_file, img_id, gt_file, gt_id, [reshape=0], [bg_bias=1])
#        ` img_file: image file(.mat) name
#        ` img_id: the var name in image file
#        ` gt_file: ground true file(.mat) name
#        ` gt_id: the var name in ground true file
#        ` reshape: reshape or not [default: 0 (not resheap)]
#        ` bg_bias: the background value in ground true image [default: 1]
#            > samples.load('PaviaU_im', 'im', 'PaviaU_gt_map', 'map', 1, 1)
#  4.normalization:
#      samples.normalization([nor_type=-1, save=[]])
#        ` nor_type: -1: normalize to [-1, 1]
#                     0: normalize to [0, 1]
#          save: a numpy array, the scale of normalization, if [] is given, a
#                numpy array (the scale) will return [default: []]
#           > save_w = samples.normalization()
#           > samples.normalization(save=save_w)
#           # if you have 2 RSI, you train a classifer with RSI1, but then
#             you want to predict RSI2, you should normalize RSI2 with the
#             scale given by RSI2:
#           > samples.normalization(save=save_w)
#  5.initial training samples
#      samples.init_sam(n_perclass, [n_test=-1, self_train=0])
#        ` n_perclass: number of samples per class
#        ` n_test: number of samples per class, [default: -1(all the rest of
#                  samples is used as test samples)]
#        ` self_train: self train or not, [default: 0]
#      samples.init_sam2(ntrain_list, nvali_list)
#        ` ntrain_list: number of train samples per class
#        ` nvali_list: number of validation samples per class, input a list of
#                      int, if the length of list is 1, every class will get
#                      the same number of samples.
#        ` The rest of samples is for test set.
#           > samples.init_sam(10)
#           > samples.init_sam(10, 20)
#           > samples.init_sam2([10, 5, 25, 30], 20)
#           > samples.init_sam2([10, 5, 25, 30], [20, 10, 50, 60])
#  6.load samples (make samples)
#      samples.load_all_sam()
#  7.save or load samples set
#      samples.save(Filename)
#        ` Save labeled samples only.
#      samples.save_all(Filename)
#        ` Save all samples.
#      samples.load(Filename)
#        ` Load labeled samples only.
#      samples.load_all(Filename)
#        ` Load all samples.
#  8.Active Learning
#      samples.selector(stype, para)
#        ` stype: AL method
#        ` para: parameters of AL method
#


import random
import h5py
import copy
import scipy
import numpy as np


def to_categorical(cla, ncls):
    _tt = []
    for _i in cla:
        _t = np.zeros(ncls, dtype=np.int32)
        _t[_i] = 1
        _tt.append(_t)
    return np.array(_tt)


class SSM:

    # initial the SSM
    def __init__(self, IMGSize, SAMN, CLAN, AUG=1, RSEED=-1):
        self.img = {'im': [],
                    'gt': [],
                    'pad': [],
                    }
        self.all = {'sample': [],
                    'label': [],
                    'label_v': [],
                    'XY': []
                    }
        self.train = {'sample': [],
                      'label': [],
                      'label_v': [],
                      'XY': []
                      }
        self.test = {'sample': [],
                     'label': [],
                     'label_v': [],
                     'XY': []
                     }
        self.candidate = {'sample': [],
                          'label': [],
                          'label_v': [],
                          'XY': []
                          }
        self.IMGX = IMGSize[0]
        self.IMGY = IMGSize[1]
        self.IMGB = IMGSize[2]
        self.SAMD = SAMN // 2
        self.SAMN = SAMN
        self.CLAN = CLAN
        self.AUG = AUG
        if RSEED != -1:
            random.seed(RSEED)

    # destructor
    def __del__(self):
        self.img.clear()
        self.all.clear()
        self.train.clear()
        self.test.clear()
        self.candidate.clear()
        del self.img
        del self.all
        del self.train
        del self.test
        del self.candidate
        try:
            self.self_sam.clear()
            self.flush.clear()
            del self.self_sam
            del self.flush
        except:
            pass
        print('SSM Destructor: Delete samples!')

    # load the image and padding
    def load(self, img_file, img_id, gt_file, gt_id, reshape=0, bg_bias=1):
        self.img['gt'] = scipy.io.loadmat(gt_file)[gt_id]
        self.img['im'] = scipy.io.loadmat(img_file)[img_id]
        self.img['gt'] = self.img['gt'] - bg_bias
        if reshape == 1:
            self.img['gt'] = self.img['gt'].T
            self.img['im'] = np.reshape(self.img['im'].T,
                                        [self.IMGX, self.IMGY, self.IMGB])
        img_im = self.img['im']
        n = self.SAMD
        # Padding
        r1 = np.repeat([img_im[0, :, :]], n, axis=0)
        r2 = np.repeat([img_im[-1, :, :]], n, axis=0)
        img_add = np.concatenate((r1, img_im, r2))
        r1 = np.reshape(img_add[:, 0, :], [self.IMGX + 2 * n, 1, self.IMGB])
        r2 = np.reshape(img_add[:, -1, :], [self.IMGX + 2 * n, 1, self.IMGB])
        r1 = np.repeat(r1, self.SAMD, axis=1)
        r2 = np.repeat(r2, self.SAMD, axis=1)
        self.img['pad'] = np.concatenate((r1, img_add, r2), axis=1)

    # normalize the sample of the padded image
    # note that normalization should do before samples-getting function, such
    # as init_sam, init_sam2 and load_all_sam.
    # a numpy array will be returned if the parameter `save` is not given,
    # which can use as a parameter for normalization
    def normalization(self, nor_type=-1, save=[]):
        img_add = self.img['pad'].astype('float')
        if save == []:
            save_w = np.zeros((2, self.IMGB))
            for i in range(self.IMGB):
                save_w[0, i] = img_add[:, :, i].min()
                img_add[:, :, i] = img_add[:, :, i] - img_add[:, :, i].min()
                save_w[1, i] = img_add[:, :, i].max()
                img_add[:, :, i] = img_add[:, :, i] / img_add[:, :, i].max()
                if nor_type == -1:
                    img_add[:, :, i] = img_add[:, :, i] * 2 - 1
            self.img['pad'] = img_add
            return save_w
        else:
            for i in range(self.IMGB):
                img_add[:, :, i] = img_add[:, :, i] - save[0, i]
                img_add[:, :, i] = img_add[:, :, i] / save[1, i]
                if nor_type == -1:
                    img_add[:, :, i] = img_add[:, :, i] * 2 - 1
            self.img['pad'] = img_add

    # initial training samples, by given number of training and test samples
    # every class get the same number of samples, no validation set.
    def init_sam(self, n_perclass, n_test=-1, self_train=0):
        if self_train:
            self.self_sam = {}
            self.self_sam['map'] = np.zeros(self.img['gt'].shape)
            self.flush = {'XY': [],
                          'label': [],
                          'sample': []
                          }
        for i in range(self.CLAN):
            c_label = list(np.array(np.where(self.img['gt'] == i)).T)
            random.shuffle(c_label)
            self.train['XY'].extend(c_label[:n_perclass])
            if n_perclass == 0:
                self.test['XY'].extend(c_label[:])
                continue
            if n_test < 0:
                self.test['XY'].extend(c_label[n_perclass:])
            else:
                self.test['XY'].extend(c_label[n_perclass:n_perclass + n_test])
        random.shuffle(self.train['XY'])
        random.shuffle(self.test['XY'])
        for i in self.train['XY']:
            self.train['sample'].append(self.get_sample(i))
            self.train['label'].append(self.get_label(i))
            if self_train:
                self.self_sam['map'][i[0]][i[1]] = 1
        for i in self.test['XY']:
            self.test['sample'].append(self.get_sample(i))
            self.test['label'].append(self.get_label(i))
        self.train['label_v'] = to_categorical(self.train['label'], self.CLAN)
        self.test['label_v'] = to_categorical(self.test['label'], self.CLAN)
        for i in self.train:
            self.train[i] = np.array(self.train[i])
            self.test[i] = np.array(self.test[i])
            if self_train:
                self.candidate[i] = np.array(self.candidate[i])
        if not self_train:
            self.candidate = copy.deepcopy(self.test)
        else:
            self.self_sam['XY'] = copy.deepcopy(self.train['XY'])
            self.self_sam['label'] = copy.deepcopy(self.train['label'])
        if n_perclass != 0:
            if self.AUG != 0:
                self.train = self.make_sample(self.train)

    # initial training samples, by given number of training and validation
    # samples for every class, the rest samples is test set.
    def init_sam2(self, ntrain_list, nvali_list):
        self.vali = {'sample': [],
                     'label': [],
                     'label_v': [],
                     'XY': []
                     }
        for i in range(self.CLAN):
            if len(ntrain_list) == self.CLAN:
                i_ = i
            else:
                i_ = 0
            if len(nvali_list) == self.CLAN:
                j_ = i
            else:
                j_ = 0
            c_label = list(np.array(np.where(self.img['gt'] == i)).T)
            random.shuffle(c_label)
            self.train['XY'].extend(c_label[:ntrain_list[i_]])
            self.vali['XY'].extend(c_label[ntrain_list[i_]:ntrain_list[i_] +
                                   nvali_list[j_]])
            self.test['XY'].extend(c_label[ntrain_list[i_] + nvali_list[j_]:])
        random.shuffle(self.train['XY'])
        random.shuffle(self.vali['XY'])
        random.shuffle(self.test['XY'])

        for i in self.train['XY']:
            self.train['sample'].append(self.get_sample(i))
            self.train['label'].append(self.get_label(i))
        for i in self.test['XY']:
            self.test['sample'].append(self.get_sample(i))
            self.test['label'].append(self.get_label(i))
        for i in self.vali['XY']:
            self.vali['sample'].append(self.get_sample(i))
            self.vali['label'].append(self.get_label(i))
        self.train['label_v'] = to_categorical(self.train['label'], self.CLAN)
        self.test['label_v'] = to_categorical(self.test['label'], self.CLAN)
        self.vali['label_v'] = to_categorical(self.vali['label'], self.CLAN)

        for i in self.train:
            self.train[i] = np.array(self.train[i])
            self.test[i] = np.array(self.test[i])
            self.vali[i] = np.array(self.vali[i])
        self.candidate = copy.deepcopy(self.test)
        if self.AUG != 0:
            self.train = self.make_sample(self.train)

    # load all pixel as samples for predicting, caution that large memory will
    # be occupied.
    def load_all_sam(self):
        self.all = {'sample': [],
                    'label': [],
                    'label_v': [],
                    'XY': []
                    }
        for i in range(self.IMGX):
            for j in range(self.IMGY):
                xy = np.array([i, j])
                self.all['XY'].append(xy)
                self.all['sample'].append(self.get_sample(xy))
                self.all['label'].append(self.get_label(xy))

    def save(self, name):
        with h5py.File(name + '_img', 'w') as h5f:
            for i in self.img:
                h5f.create_dataset(i, data=self.img[i])

        with h5py.File(name + '_train', 'w') as h5f:
            for i in self.train:
                h5f.create_dataset(i, data=self.train[i])

        with h5py.File(name + '_candidate', 'w') as h5f:
            for i in self.candidate:
                h5f.create_dataset(i, data=self.candidate[i])

        with h5py.File(name + '_test', 'w') as h5f:
            for i in self.test:
                h5f.create_dataset(i, data=self.test[i])

    def save_all(self, name):
        with h5py.File(name, 'w') as h5f:
            for i in self.all:
                h5f.create_dataset(i, data=self.all[i])

    def load_file(self, name):
        with h5py.File(name + '_img', 'r') as h5f:
            for i in h5f:
                self.img[i] = h5f[i][:]

        with h5py.File(name + '_train', 'r') as h5f:
            for i in h5f:
                self.train[i] = h5f[i][:]

        with h5py.File(name + '_candidate', 'r') as h5f:
            for i in h5f:
                self.candidate[i] = h5f[i][:]

        with h5py.File(name + '_test', 'r') as h5f:
            for i in h5f:
                self.test[i] = h5f[i][:]

    def load_all_file(self, name):
        with h5py.File(name, 'r') as h5f:
            for i in h5f:
                self.all[i] = h5f[i][:]

    def loop_sample(self, no_time, batch_size):
        samn = len(self.train['sample'])
        _s = no_time * batch_size % samn
        _t = _s + batch_size
        _sam = self.train['sample'][_s:_t]
        _lv = self.train['label_v'][_s:_t]
        _lab = self.train['label'][_s:_t]
        _delta = _t - samn
        if _delta > 0:
            _sam = np.concatenate([_sam, self.train['sample'][:_delta]])
            _lv = np.concatenate([_lv, self.train['label_v'][:_delta]])
            _lab = np.concatenate([_lab, self.train['label'][:_delta]])
        return _sam, _lv, _lab

    # AL-selector
    def selector(self, stype, para):
        n = para[0]
        try:
            predict = para[1]
        except IndexError:
            pass
        if stype == 'RS':
            indexs = self.RS_sample(n)
        elif stype == 'BT':
            indexs = self.BT_sample(predict, n)
        elif stype == 'SM':
            indexs = self.SM_sample(predict, n)
        elif stype == 'MI':
            indexs = self.MI_sample(predict, n)
        elif stype == 'MBT':
            indexs = self.MBT_sample(predict, n)
        elif stype == 'entropy':
            indexs = self.entropy_sample(predict, n)
        self.add_sample(indexs)

    # filp, mirror
    def make_sample(self, origin):
        sample = origin['sample']
        label = origin['label']
        label_v = origin['label_v']
        XY = origin['XY']
        a = np.flip(sample, 1)
        b = np.flip(sample, 2)
        c = np.flip(b, 1)
        new = {}
        new['sample'] = np.concatenate((a, b, c, sample), axis=0)
        new['label'] = np.concatenate((label, label, label, label), axis=0)
        new['label_v'] = np.concatenate(
                (label_v, label_v, label_v, label_v),
                axis=0)
        new['XY'] = np.concatenate((XY, XY, XY, XY), axis=0)
        return new

    # return a sample(N*N sub-image), by given XY
    def get_sample(self, xy):
        d = self.SAMD
        x = xy[0]
        y = xy[1]
        try:
            self.img['im'][x][y]
        except IndexError:
            return []
        # considering the padded pixel, xy should add d
        x += d
        y += d
        sam = self.img['pad'][(x - d): (x + d + 1), (y - d): (y + d + 1)]
        return np.array(sam)

    # return lebel of a sample, by given XY
    def get_label(self, xy):
        return self.img['gt'][xy[0]][xy[1]]

    # add samples to training set from candidate set
    def add_sample(self, indexs):
        new = {}
        for i in self.train:
            new[i] = self.candidate[i][indexs]
            self.candidate[i] = np.delete(self.candidate[i], indexs, axis=0)
        if self.AUG != 0:
            new = self.make_sample(new)
        for i in self.train:
            self.train[i] = np.append(self.train[i], new[i], 0)

    # Active Learning Algorithm
    def RS_sample(self, n):
        cand_sam = self.candidate
        index = list(range(len(cand_sam['sample'])))
        random.shuffle(index)
        return index[:n]

    def BT_sample(self, predict, n):
        index = np.argsort(-predict)
        diff = np.zeros((len(index), 2))
        for i in range(len(index)):
            diff[i, 0] = predict[i, index[i, 0]]
            diff[i, 1] = predict[i, index[i, 1]]
        diff2 = diff[:, 0] - diff[:, 1]
        index = np.argsort(diff2)
        return index[:n]

    # sort by max probability
    def new_sample(self, predict, n):
        probability = predict.max(axis=1)
        index = np.argsort(probability)
        return index[:n]

    # cos
    def cos_sample(self, predict, n):
        probability = predict.max(axis=1)
        index = np.argsort(probability)
        return index[:n]

    def SM_sample(self, predict, n):
        index = np.argsort(-predict)
        diff = np.zeros((len(index), 2))
        for i in range(len(index)):
            diff[i, 0] = predict[i, index[i, 0]]
            diff[i, 1] = predict[i, index[i, 1]]
        diff2 = diff[:, 1]
        index = np.argsort(-diff2)
        return index[:n]

    def MBT_sample(self, predict, n):
        predictlabel = np.argmax(predict, axis=1)
        total_index = []
        total_p2 = []
        for i in range(9):
            tmpindex = np.where(predictlabel == i)
            tmpindex = tmpindex[0].T
            classpredict = predict[tmpindex, :]
            classpredict[:, i] = 0.0
            maxx = np.max(classpredict, axis=1)
            index2 = np.argsort(-maxx)
            total_p2.append(maxx[index2[:2]])
            total_index.append(tmpindex[index2[:2]])
        total_p2 = np.array(total_p2).reshape(18)
        total_index = np.array(total_index).reshape(18)
        index3 = np.argsort(-total_p2.T)[:10]
        index = total_index[index3]
        return index[:n]

    def entropy_sample(self, predict, n):
        etp = scipy.stats.entropy(predict.T)
        index = np.argsort(-etp)
        return index[:n]

    # Self-Train Model
    def add_self_sam(self, x, y, label):
        self_map = self.self_sam['map']
        try:
            self_map[x][y]
            if (x < 0) | (y < 0):
                raise IndexError
        except IndexError:
            return
        if (not self_map[x][y]):
            XY = np.array([x, y])
            self.flush['XY'].append(XY)
            self.flush['label'].append(label)
            self.flush['sample'].append(self.get_sample(XY))

    def st_flush(self):
        l = len(self.self_sam['label'])
        for i in self.flush:
            del self.flush[i]
            self.flush[i] = []
        for i in range(l):
            xy = self.self_sam['XY'][i]
            label = self.self_sam['label'][i]
            self.add_self_sam(xy[0] - 1, xy[1], label)
            self.add_self_sam(xy[0] + 1, xy[1], label)
            self.add_self_sam(xy[0], xy[1] + 1, label)
            self.add_self_sam(xy[0], xy[1] - 1, label)
        for i in self.flush:
            self.flush[i] = np.array(self.flush[i])
        return self.flush['sample']

    def st_merge(self, predict):
        indexs = np.where(self.flush['label'] != predict)
        indexs = list(indexs[0])
        for i in self.flush:
            self.flush[i] = np.delete(self.flush[i], indexs, axis=0)
            try:
                self.candidate[i] = np.append(
                        self.candidate[i],
                        self.flush[i],
                        0)
            except ValueError:
                self.candidate[i] = self.flush[i]
        self.candidate['label_v'] = to_categorical(
                self.candidate['label'],
                self.CLAN)
        for i in ['XY', 'label']:
            self.self_sam[i] = np.append(self.self_sam[i], self.flush[i], 0)
        for i in self.flush['XY']:
            self.self_sam['map'][i[0]][i[1]] = 1
