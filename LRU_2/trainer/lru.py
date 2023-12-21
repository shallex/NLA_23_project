from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from .utils import *
from .loggers import *
from .base import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import json
import numpy as np
from abc import *
from pathlib import Path

def compute_accuracy(logits, label):
    return np.argmax(logits.detach().cpu().numpy()) == label.detach().cpu().numpy()


class LRUTrainer(BaseTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root, use_wandb):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root, use_wandb)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

    def calculate_loss(self, batch):
        seqs, labels = batch
        seqs = seqs.float()
        # print(f"{seqs.shape=}")
        
        if self.args.dataset_code != 'xlong':
            logits = self.model(seqs)
            # print(f"{logits.shape=}")
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            loss = self.ce(logits, labels)
        else:
            logits, labels_ = self.model(seqs, labels=labels)
            logits = logits.view(-1, logits.size(-1))
            labels_[labels==0] = 0
            labels_ = labels_.view(-1)
            loss = self.ce(logits, labels_)
        return loss

    def calculate_metrics(self, batch):
        seqs, labels = batch
        # print(labels)
        seqs = seqs.float()
        
        if self.args.dataset_code != 'xlong':
            scores = self.model(seqs)
            # print(scores)
        #     B, L = scores.shape
        #     for i in range(L):
        #         scores[torch.arange(scores.size(0)), seqs[:, i]] = -1e9
        #     scores[:, 0] = -1e5  # padding
        # else:
        #     scores, labels = self.model(seqs, labels=labels)
        #     scores = scores[:, -1, :]
        
        metrics = absolute_recall_mrr_ndcg_for_ks(scores, labels.view(-1), self.metric_ks)
        acc = np.mean(compute_accuracy(scores, labels))
        print(f"Accuracy: {acc}")
        metrics["Accuracy"] = acc
        return metrics