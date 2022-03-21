# coding=utf-8
# @FileName     :entity_server.py
# @DateTime     :2022/3/9 17:04
# @Author       :Haoguo Feng

import pickle
import tensorflow as tf
from utils import create_model_for_bert, get_logger
from model.bert_lstm_crf import BertLSTMCRF
from train.train_gpu import load_config
from train.train_helper import get_train_args
from loader import input_from_line

train_args = get_train_args()
config = load_config(train_args.config_file)
logger = get_logger(train_args.log_file)
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True

with open(train_args.map_file, "rb") as f:
    tag_to_id, id_to_tag = pickle.load(f)
sess = tf.compat.v1.Session(config=tf_config)
model = create_model_for_bert(sess, BertLSTMCRF, train_args.ckpt_path, config, logger)


def extract_entity(chstr):
    result = model.evaluate_line(sess,
                                 input_from_line(chstr, max_seq_length=train_args.max_seq_len, tag_to_id=tag_to_id),
                                 id_to_tag=id_to_tag)
    loc = set()
    per = set()
    org = set()
    for d in result['entities']:
        if d['type'] == 'LOC':
            loc.add(d['word'])
        if d['type'] == 'PER':
            per.add(d['word'])
        if d['type'] == 'ORG':
            org.add(d['word'])
    return list(per), list(loc), list(org)


if __name__ == "__main__":
    run_code = 0
