# -*- coding: utf-8 -*-
import dataset
import keys_keras
import numpy as np
import torch

characters = keys_keras.alphabet[:]
from model import get_model

nclass = len(characters) + 1

trainroot = '../data/lmdb/train'
valroot = '../data/lmdb/val'
batchSize = 32
workers = 0
imgH = 32
imgW = 256
keep_ratio = False
random_sample = False
LEARNING_RATE = 0.01




def one_hot(text, length=10, characters=characters):
    label = np.zeros(length)
    # print(type(text))
    for i, char in enumerate(text):
        index = characters.find(char)
        if index == -1:
            index = characters.find(u' ')
        # print(i,char,length)
        # if i < length:
        label[i] = index
    return label


n_len = 10


def gen(loader, flag='train'):
    while True:
        i = 0
        n = len(loader)
        for X, Y in loader:
            X = X.numpy()
            X = X.reshape((-1, imgH, imgW, 1))
            if flag == 'test':
                Y = Y.numpy()

            Y = np.array(Y)
            Length = int(imgW / 4) - 1
            batchs = X.shape[0]
            # Y = Y.numpy()
            if i > n - 1:
                i = 0
                break

            yield [
                X, Y,
                np.ones(batchs) * int(Length),
                np.ones(batchs) * n_len
            ], np.ones(batchs)


if random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, batchSize)
else:
    sampler = None
train_dataset = dataset.lmdbDataset(root=trainroot, target_transform=one_hot)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batchSize,
    shuffle=True,
#     sampler=sampler,
#     num_workers=int(workers),
    collate_fn=dataset.alignCollate(
        imgH=imgH, imgW=imgW, keep_ratio=keep_ratio))

test_dataset = dataset.lmdbDataset(
    root=valroot,
    transform=dataset.resizeNormalize((imgW, imgH)),
    target_transform=one_hot)

test_loader = torch.utils.data.DataLoader(
    test_dataset, shuffle=True, batch_size=batchSize, num_workers=int(workers))

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

model, basemodel = get_model(height=imgH, nclass=nclass,learning_rate=LEARNING_RATE)
import os

if os.path.exists(
        '/mnt/wuwenhui/git_ocr_project/CHINESE-OCR/model/my_model_keras.h5'
):
    basemodel.load_weights(
        '/mnt/wuwenhui/git_ocr_project/CHINESE-OCR/model/my_model_keras.h5'
    )

    ##注意此处保存的是model的权重
#     checkpointer = ModelCheckpoint(
#         filepath=
#          '/mnt/wuwenhui/git_ocr_project/CHINESE-OCR/save_model/my_model_keras.h5",
#         monitor='val_loss',
#         verbose=0,
#         save_weights_only=False,
#         save_best_only=True)

rlu = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=1,
    verbose=0,
    mode='auto',
    epsilon=0.0001,
    cooldown=0,
    min_lr=0)

model.fit_generator(
    gen(train_loader, flag='train'),
    steps_per_epoch=200,
    epochs=200,)
#     validation_data=gen(test_loader, flag='test'),
# #     callbacks=[checkpointer, rlu],
#     validation_steps=72)