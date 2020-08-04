from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import six
import collections
import unicodedata
import tensorflow as tf

def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
    if not init_checkpoint:
        return

    m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)

    model_name = m.group(1)

    lower_models = [
        "uncased_L-24_H-1024_A-16", "uncased_L-12_H-768_A-12",
        "multilingual_L-12_H-768_A-12", "chinese_L-12_H-768_A-12"
    ]

    cased_models = [
        "cased_L-12_H-768_A-12", "cased_L-24_H-1024_A-16",
        "multi_cased_L-12_H-768_A-12"
    ]

    is_bad_config = False

    if model_name in lower_models and not do_lower_case:
        is_bad_config = True
        actual_flag = "False"
        case_name = "lowercased"
        oppsite_flag = "True"

    if model_name in cased_models and do_lower_case:
        is_bad_config = True
        actual_flag = "True"
        case_name = "cased"
        opposite_flag = "False"

    if is_bad_config:
        raise ValueError("You passed in `--do_lower_case=%s` with `--init_checkpoint=%s`. "
        "However, `%s` seems to be a %s model, so you "
        "should pass in `--do_lower_case=%s` so that the fine-tuning matches "
        "how the model was pre-training. If this error is wrong, please "
        "just comment out this check." % (actual_flag, init_checkpoint,
                                          model_name, case_name, opposite_flag))
    