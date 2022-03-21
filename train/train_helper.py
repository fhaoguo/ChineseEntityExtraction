# coding=utf-8
# @FileName     :main_bert.py
# @DateTime     :2022/2/14 14:28
# @Author       :Haoguo Feng

import os
import argparse


def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, help='train?', required=False, default=True)
    # configurations for the model
    parser.add_argument('--clean', type=bool, help='clean?', required=False, default=True)
    parser.add_argument('--batch_size', type=int, help='batch size', required=False, default=256)
    parser.add_argument("--seg_dim", type=int, help="Embedding size for segmentation, 0 if not used",
                        required=False, default=20)
    parser.add_argument("--char_dim", type=int, help="Embedding size for characters",
                        required=False, default=100)
    parser.add_argument("--lstm_dim", type=int, help="Num of hidden units in LSTM",
                        required=False, default=200)
    parser.add_argument("--tag_schema", type=str, help="tagging schema iobes or iob",
                        required=False, default="iob")
    # configurations for training
    parser.add_argument("--clip", type=float, help="Gradient clip",
                        required=False, default=0.25)
    parser.add_argument("--dropout", type=float, default=0.5, required=False, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=0.001, required=False, help="Initial learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", required=False, help="Optimizer for training")
    parser.add_argument("--zeros", type=bool, default=False, help="Wither replace digits with zero", required=False)
    parser.add_argument("--lower", type=bool, default=True, help="Wither lower case", required=False)

    parser.add_argument("--max_seq_len", type=int, default=128, help="max sequence length for bert", required=False)
    parser.add_argument("--max_epoch", type=int, default=180, help="maximum training epochs", required=False)
    parser.add_argument("--steps_check", type=int, default=100, help="steps per checkpoint", required=False)
    parser.add_argument("--ckpt_path", type=str, default="ckpt", help="Path to save model", required=False)
    parser.add_argument("--summary_path", type=str, default="summary", help="Path to store summaries", required=False)
    parser.add_argument("--log_file", type=str, default="train.log", help="File for log", required=False)
    parser.add_argument("--map_file", type=str, default="maps.pkl", help="file for maps", required=False)
    parser.add_argument("--vocab_file", type=str, default="vocab.json", help="File for vocab", required=False)
    parser.add_argument("--config_file", type=str, default="config_file", help="File for config", required=False)
    parser.add_argument("--script", type=str, default="conlleval", help="evaluation script", required=False)
    parser.add_argument("--result_path", type=str, default="result", help="Path for results", required=False)
    parser.add_argument("--train_file", type=str, default=os.path.join("data", "example.train"),
                        help="Path for train data", required=False)
    parser.add_argument("--dev_file", type=str, default=os.path.join("data", "example.dev"),
                        help="Path for dev data", required=False)
    parser.add_argument("--test_file", type=str, default=os.path.join("data", "example.test"),
                        help="Path for test data", required=False)
    args = parser.parse_args()
    return args
