# coding=utf-8
# @FileName     :predict_bert.py
# @DateTime     :2022/2/14 15:33
# @Author       :Haoguo Feng

import pickle
import tensorflow as tf
from utils import create_model_for_bert, get_logger
from model.bert_lstm_crf import BertLSTMCRF
from loader import input_from_line
from train.train_gpu import load_config
from re_expr import detect_chinese
from train.train_helper import get_train_args


def predict(_):
    train_args = get_train_args()
    config = load_config(train_args.config_file)
    logger = get_logger(train_args.log_file)
    # limit GPU memory
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(train_args.map_file, "rb") as f:
        tag_to_id, id_to_tag = pickle.load(f)
    with tf.compat.v1.Session(config=tf_config) as sess:
        model = create_model_for_bert(sess, BertLSTMCRF, train_args.ckpt_path, config, logger)
        while True:
            line = input("请输入测试句子:")
            if line.lower() in ["q", "e", "quit", "exit"]:
                break
            if detect_chinese(line) is None:
                logger.info("非法输入，请重新输入或按《q》键退出！")
                continue
            result = model.evaluate_line(sess,
                                         input_from_line(
                                             line, max_seq_length=train_args.max_seq_len, tag_to_id=tag_to_id),
                                         id_to_tag)
            logger.info(result)
            print(result)


if __name__ == '__main__':
    tf.compat.v1.app.run(predict)
