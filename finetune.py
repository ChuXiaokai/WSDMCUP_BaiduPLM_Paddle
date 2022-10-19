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
paddle.set_device("gpu:0")
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

# 优化器设置
scheduler = get_linear_schedule_with_warmup(config.lr, config.warmup_steps,
                                        config.max_steps)
decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]

optimizer = paddle.optimizer.AdamW(
    learning_rate=scheduler,
    parameters=model.parameters(),
    weight_decay=config.weight_decay,
    apply_decay_param_fun=lambda x: x in decay_params,
    grad_clip=nn.ClipGradByNorm(clip_norm=0.5)
)
criterion = nn.BCEWithLogitsLoss()

vaild_annotate_dataset = TestDataset(config.valid_annotate_path, max_seq_len=config.max_seq_len, data_type='finetune')
vaild_annotate_loader = DataLoader(vaild_annotate_dataset, batch_size=config.eval_batch_size) 
test_annotate_dataset = TestDataset(config.test_annotate_path, max_seq_len=config.max_seq_len, data_type='annotate')
test_annotate_loader = DataLoader(test_annotate_dataset, batch_size=config.eval_batch_size) 

idx = 0
for _ in range(config.finetune_epoch):
    for valid_data_batch in vaild_annotate_loader:
        model.train()
        optimizer.clear_grad()
        src_input, src_segment, src_padding_mask, label = valid_data_batch
        score = model(
            src=src_input, 
            src_segment=src_segment, 
            src_padding_mask=src_padding_mask, 
        )
        ctr_loss = criterion(score, paddle.to_tensor(label, dtype=paddle.float32))
        ctr_loss.backward()
        optimizer.step()
        scheduler.step()

        if idx % config.log_interval == 0:
            print(f'{idx:5d}th step | loss {ctr_loss.item():5.6f}')


        if idx % config.eval_step == 0:
            model.eval()
            # ------------   evaluate on annotated data -------------- # 
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

            result_dict_ann = evaluate_all_metric(
                qid_list=vaild_annotate_dataset.total_qids, 
                label_list=vaild_annotate_dataset.total_labels, 
                score_list=total_scores, 
                freq_list=vaild_annotate_dataset.total_freqs
            )
            print(
                f'{idx}th step valid annotate | '
                f'dcg@10: all {result_dict_ann["all_dcg@10"]:.6f} | '
                f'high {result_dict_ann["high_dcg@10"]:.6f} | '
                f'mid {result_dict_ann["mid_dcg@10"]:.6f} | '
                f'low {result_dict_ann["low_dcg@10"]:.6f} | '
                f'pnr {result_dict_ann["pnr"]:.6f}'
            )
            if idx % config.save_step == 0 and idx > 0:
                paddle.save(model.state_dict(),
                        'save_model/save_steps{}_{:.5f}.model'.format(idx, result_dict_ann['pnr'])
                )
        idx += 1
