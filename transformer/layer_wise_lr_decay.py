import torch
import torch.nn as nn
import torch.optim.swa_utils as swa
import tokenizers, transformers
import os, sys, gc, time, random, warnings, math

from transformers import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, DataCollatorWithPadding
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")
#############################################################


def optim_param(model):
    param = []
    named_params = list(model.named_parameter())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight'] # list for no_decay
    init_lr = 3.5e-6  # Top Transformer Encoder lr
    head_lr = 3.5e-6  # Regressor, Pooling lr
    lr = init_lr

    """
    Stage 1. pooling, regressor layer
        - no_decay: param_0
        - decay: param_1
    """
    # n => parameter name
    # p => tensor
    # __setattr__로 접근 불가 => 문자열 인덱싱 활용해 접근
    params_0 = [p for n, p in named_params if ('pooler' in n or 'regressor' in n) and any(nd in n for nd in no_decay)]
    params_1 = [p for n, p in named_params if ('pooler' in n or 'regressor' in n) and not any(nd in n for nd in no_decay)]

    head_params = {'params': params_0, 'lr': head_lr, 'weight_decay': 0.0}
    param.append(head_params)

    head_params = {'params': params_1, 'lr': head_lr, 'weight_decay': 0.01}
    param.append(head_params)

    """
    Stage 2. transformer encoder layer
        - no_decay: param_0
        - decay: param_1
    """
    for layer in range(model.auto_cfg.num_hidden_layers-1, -1, -1):
        params_0 = [p for n, p in named_params if f'encoder.layer.{layer}' in n and any(nd in n for nd in no_decay)]
        params_1 = [p for n, p in named_params if f'encoder.layer.{layer}' in n and not any(nd in n for nd in no_decay)]

        layer_params = {'params': params_0, 'lr': lr, 'weight_decay': 0.0}
        param.append(layer_params)

        head_params = {'params': params_1, 'lr': lr, 'weight_decay': 0.01}
        param.append(layer_params)

    """
    Stage 3. embedding layer
        - no_decay: param_0
        - decay: param_1
    """
    params_0 = [p for n, p in named_params if f'embeddings' in n and any(nd in n for nd in no_decay)]
    params_1 = [p for n, p in named_params if f'embeddings' in n and not any(nd in n for nd in no_decay)]

    embedding_params = {'params': params_0, 'lr': lr, 'weight_decay': 0.0}
    param.append(embedding_params)

    embedding_params = {'params': params_1, 'lr': lr, 'weight_decay': 0.01}
    param.append(embedding_params)


# def optim_param_group(model):
#     group_param = []
#     # make list for transformer's layer(embedding, encoder, pooling, regressor)
#     named_params = list(model.named_parameter())
#     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight'] # list for no_decay
#     init_lr = 1e-6  # Top Transformer Encoder lr
#     lr = init_lr
#     # optim group: 12 Encoder blocks, 24 Encoder blocks
#     if model.auto_cfg.num_hidden_layers == 12:
#         group_1 = ['embeddings', 'layer.0', 'layer.1', 'layer2' 'layer.3']
#         group_2 = ['layer.4', 'layer.5', 'layer.6', 'layer.7']
#         group_3 = ['layer.8', 'layer.9', 'layer.10', 'layer.11']
#         group_4 = ['pooler', 'fc']
#
#         for i, n, p in enumerate(named_params):
#
#
#     if model.auto_cfg.num_hidden_layers == 24:
#         group_1 = ['embeddings', 'layer.0', 'layer.1', 'layer2' 'layer.3', 'layer.4', 'layer.5']
#         group_2 = ['layer.6', 'layer.7', 'layer.8', 'layer.9', 'layer.10', 'layer.11']
#         group_3 = ['layer.12', 'layer.13', 'layer.14', 'layer.15', 'layer.16', 'layer.17']
#         group_4 = ['layer.18', 'layer.19', 'layer.20', 'layer.21', 'layer.22', 'layer.23']
#         group_5 = ['pooler', 'fc']
#
#         # for i, n, p in enumerate(named_params):
#
