# coding=utf-8
# @FileName     :main_bert.py
# @DateTime     :2022/2/14 14:28
# @Author       :Haoguo Feng

import os
import pickle
import numpy as np
import tensorflow as tf
from collections import OrderedDict

from data_utils import BatchManager
from loader import load_sentences
from loader import prepare_dataset
from loader import tag_mapping
from model.bert_lstm_crf import BertLSTMCRF
from utils import get_logger, make_path, create_model_for_bert, save_model, print_config, save_config, \
    load_config, test_ner
from train.train_helper import get_train_args


FLAGS = get_train_args()
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]

abs_path = ''


# config for the model
def config_model(tag_to_id):
    config = OrderedDict()
    config["num_tags"] = len(tag_to_id)
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size
    config['max_seq_len'] = FLAGS.max_seq_len

    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower

    config["max_epoch"] = FLAGS.max_epoch
    config["steps_check"] = FLAGS.steps_check
    return config


def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def train():
    # load data sets
    train_sentences = load_sentences(abs_path + FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
    dev_sentences = load_sentences(abs_path + FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
    test_sentences = load_sentences(abs_path + FLAGS.test_file, FLAGS.lower, FLAGS.zeros)

    # create maps if not exist
    if not os.path.isfile(abs_path + FLAGS.map_file):
        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        with open(abs_path + FLAGS.map_file, "wb") as f:
            pickle.dump([tag_to_id, id_to_tag], f)
    else:
        with open(abs_path + FLAGS.map_file, "rb") as f:
            tag_to_id, id_to_tag = pickle.load(f)

    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(
        train_sentences, FLAGS.max_seq_len, tag_to_id, FLAGS.lower
    )
    dev_data = prepare_dataset(
        dev_sentences, FLAGS.max_seq_len, tag_to_id, FLAGS.lower
    )
    test_data = prepare_dataset(
        test_sentences, FLAGS.max_seq_len, tag_to_id, FLAGS.lower
    )
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), len(dev_data), len(test_data)))

    train_manager = BatchManager(train_data, FLAGS.batch_size, "bert")
    dev_manager = BatchManager(dev_data, FLAGS.batch_size, "bert")
    test_manager = BatchManager(test_data, FLAGS.batch_size, "bert")
    # make path for store log and model if not exist
    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(tag_to_id)
        save_config(config, FLAGS.config_file)

    log_path = os.path.join(abs_path + "log", FLAGS.log_file)
    logger = get_logger(log_path)
    print_config(config, logger)

    # limit GPU memory
    tf_config = tf.compat.v1.ConfigProto(
        log_device_placement=False,inter_op_parallelism_threads=2,intra_op_parallelism_threads=2,allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    logger.info("start training ...")
    with tf.compat.v1.Session(config=tf_config) as sess:
        model = create_model_for_bert(sess, BertLSTMCRF, FLAGS.ckpt_path, config, logger)
        loss = []
        for i in range(FLAGS.max_epoch):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, NER loss:{:>9.6f}".format(
                        iteration, step % steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
            if best:
                save_model(sess, model, FLAGS.ckpt_path, logger, global_steps=step)
        
        evaluate(sess, model, "test", test_manager, id_to_tag, logger)


if __name__ == "__main__":
    run_code = 0
