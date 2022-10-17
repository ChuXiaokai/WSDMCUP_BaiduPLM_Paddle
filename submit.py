# -*- encoding: utf-8 -*-
'''
@Time    :   2022/06/12 14:49:28
@Author  :   Chu Xiaokai 
@Contact :   xiaokaichu@gmail.com
'''
import numpy as np
import warnings
import sys
from metrics import *
from Transformer4Ranking.model import *
from paddle.io import DataLoader
from dataloader import *
from args import config

random.seed(config.seed+1)
random.seed(config.seed)
np.random.seed(config.seed)
paddle.set_device("gpu:3")
paddle.seed(config.seed)
print(config)
exp_settings = config.exp_settings

model = TransformerModel(
    ntoken=config.ntokens, 
    hidden=config.emb_dim, 
    nhead=config.nhead, 
    nlayers=config.nlayers, 
    dropout=config.dropout,
    mode='finetune'
)

# load pretrained model
if config.init_parameters != "":
    print('load warm up model ', config.init_parameters)
    ptm = paddle.load(config.init_parameters)
    for k, v in model.state_dict().items():
        if not k in ptm:    
            pass
            print("warning: not loading " + k)
        else:
            print("loading " + k)
            v.set_value(ptm[k])

test_annotate_dataset = TestDataset(config.test_annotate_path, max_seq_len=config.max_seq_len, data_type='annotate')
test_annotate_loader = DataLoader(test_annotate_dataset, batch_size=config.eval_batch_size) 
# evaluate
total_scores = []

for test_data_batch in test_annotate_loader:
    src_input, src_segment, src_padding_mask, label = test_data_batch
    score = model(
        src=src_input, 
        src_segment=src_segment, 
        src_padding_mask=src_padding_mask, 
    )
    score = score.cpu().detach().numpy().tolist()
    total_scores += score

with open(config.result_path, "w") as f:
    f.writelines("\n".join(map(str, total_scores)))

