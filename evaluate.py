# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# python evaluate.py --crosslingual --src_lang en --tgt_lang es --src_emb data/wiki.en-es.en.vec --tgt_emb data/wiki.en-es.es.vec

import os
import argparse
from collections import OrderedDict

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator
import pandas as pd
# main
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
# data
parser.add_argument("--src_lang", type=str, default="", help="Source language")
parser.add_argument("--tgt_lang", type=str, default="", help="Target language")
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default="", help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default="", help="Reload target embeddings")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")
parser.add_argument("--model_name",type=str, default="", help="Model name")

# parse parameters
params = parser.parse_args()

# check parameters
assert params.src_lang, "source language undefined"
assert os.path.isfile(params.src_emb)
assert not params.tgt_lang or os.path.isfile(params.tgt_emb)
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)

# build logger / model / trainer / evaluator
logger = initialize_exp(params)
src_emb, tgt_emb, mapping, _ = build_model(params, False)
trainer = Trainer(src_emb, tgt_emb, mapping, None, params)
evaluator = Evaluator(trainer)

# run evaluations
to_log = OrderedDict({'n_iter': 0})
log_export  = {
    "name" : params.model_name
}
evaluator.monolingual_wordsim(to_log,log_export)
evaluator.monolingual_wordanalogy(to_log,log_export)
if params.tgt_lang:
    # evaluator.crosslingual_wordsim(to_log)
    evaluator.word_translation(to_log)
    # evaluator.sent_translation(to_log)
    evaluator.dist_mean_cosine(to_log)

log_export['nn-1'] = to_log['precision_at_1-nn']
log_export['nn-5'] = to_log['precision_at_5-nn']
log_export['nn-10'] = to_log['precision_at_10-nn']
log_export['csls-1'] = to_log['precision_at_1-csls_knn_10']
log_export['csls-5'] = to_log['precision_at_5-csls_knn_10']
log_export['csls-10'] = to_log['precision_at_10-csls_knn_10']
log_export['mean-cosine-nn']= to_log['mean_cosine-nn-S2T-10000']
log_export['mean-cosine-csls']= to_log['mean_cosine-csls_knn_10-S2T-10000']
print("=" * 30)
print(params.model_name + ' Result')
print(log_export)
print("=" * 30)
df = pd.DataFrame([log_export])
df.to_csv(f"/kaggle/working/{params.model_name}.csv")
