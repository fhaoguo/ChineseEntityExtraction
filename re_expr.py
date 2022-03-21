# coding=utf-8
# @FileName     :re_expr.py
# @DateTime     :2022/2/11 16:38
# @Author       :Haoguo Feng

import re


def startwith_non_chinese(line):
    ch_str = u"^[\u4e00-\u9fff]+"
    pattern = re.compile(ch_str)
    return not pattern.match(line)


def detect_sentence_border(line):
    sent_border = u"^[(（）)<>《》.。？?;；！!,，]+"
    pattern = re.compile(sent_border)
    return pattern.search(line)


def detect_chinese(line):
    ch_str = u"([\u4e00-\u9fff]+)"
    pattern = re.compile(ch_str)
    return pattern.search(line)


if __name__ == "__main__":
    run_code = 0
