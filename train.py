# coding=utf-8
# @FileName     :predict_bert.py
# @DateTime     :2022/2/14 15:33
# @Author       :Haoguo Feng

import os
import tensorflow as tf
from utils import clean
from train.train_helper import get_train_args


def main(_):
    from train.train_gpu import train
    train_args = get_train_args()
    train_args.train = True
    train_args.clean = True
    clean(train_args)
    tf.keras.backend.clear_session()
    tf.config.optimizer.set_jit(True)  # Enable XLA
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    train()


if __name__ == '__main__':
    tf.compat.v1.app.run(main)
