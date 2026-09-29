# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from sklearn import preprocessing
# from ret_benchmark.losses.registry import LOSS
import random
import pdb
# random.seed(2)
# @LOSS.register('ms_loss')
class MultiSimilarityLoss(nn.Module):
    def __init__(self, scale_pos, scale_neg): #初始化損失函數的超參數
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 1
        self.margin = 0.3

        self.scale_pos = scale_pos
        self.scale_neg = scale_neg

    def forward(self, feats, attrs, labels, attr_mask):

        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
 
        feats = torch.nn.functional.normalize(feats, p=2.0, dim=1, eps=1e-12, out=None) #特徵歸一化：將特徵向量 L2 歸一化，確保相似度計算在單位球面上
        # print(feats)
        sim_mat = torch.matmul(feats, torch.t(feats)) #計算相似度矩陣：sim_mat 形狀為 (batch_size, batch_size)，每個元素是兩個樣本的特徵余弦相似度
        if attrs != None: #若啟用 attr_mask，將特徵相似度與屬性相似度相乘，增強與屬性相關的相似度
            # print('==================')
            attrs = torch.nn.functional.normalize(attrs, p=2.0, dim=1, eps=1e-12, out=None)
            att_sim_mat = torch.matmul(attrs, torch.t(attrs))
        epsilon = 1e-5
        loss = list()
        if attr_mask == 1:
            sim_mat = sim_mat * att_sim_mat
            # print('here')
      
        labels = labels*2 - 1
        labels = labels.float()
        label_mat = torch.matmul(labels.unsqueeze(1), labels.unsqueeze(1).t())

  
        label_mat = (label_mat + 1) / 2 #label_mat 計算後，相同標籤的位置為 1，不同為 0
        # label_mat = 1-label_mat
        pos_pair_ = sim_mat * label_mat

        tmp = pos_pair_< 1 - epsilon

        pos_pair_ = pos_pair_*tmp.long()

        neg_label_mat = 1 - label_mat
        neg_label_tmp_mat = 1 - label_mat*tmp.long()
        label_tmp_mat = label_mat*tmp.long()

        neg_pair_ = sim_mat * neg_label_mat

        min_pos = torch.min((pos_pair_+neg_label_tmp_mat*torch.max(pos_pair_)),dim = 1)[0]
        max_neg = torch.max(neg_pair_-label_mat*torch.max(neg_pair_),dim = 1)[0]

        min_pos = min_pos.unsqueeze(1)
        max_neg = max_neg.unsqueeze(1)

        min_pos_ = min_pos.expand(sim_mat.shape[0], sim_mat.shape[1])
        max_neg_ = max_neg.expand(sim_mat.shape[0], sim_mat.shape[1])
        neg_pair_tmp = neg_pair_ + self.margin > min_pos_ 

        neg_pair_tmp2 = neg_label_mat != 0
        pos_pair_tmp = pos_pair_ - self.margin < max_neg_ 
        pos_pair_tmp2 = label_mat != 0

        neg_pair_tmp = neg_pair_tmp.long() * neg_pair_tmp2.long()
        pos_pair_tmp = pos_pair_tmp.long() * pos_pair_tmp2.long()
        neg_flase = sim_mat.shape[1] - neg_label_mat.sum(dim=1)
        pos_flase = sim_mat.shape[1] - label_tmp_mat.sum(dim=1)
        pos_pair = pos_pair_ * pos_pair_tmp
        neg_pair = neg_pair_ * neg_pair_tmp
  
        pos_pair += neg_label_tmp_mat * self.thresh
        neg_pair += label_mat * self.thresh

        pos_sum = 1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh)), dim=1) - pos_flase
        pos_sum = torch.clamp(pos_sum, min=1e-12)
        pos_loss = 1.0 / self.scale_pos * torch.log(pos_sum)

        #pos_loss = 1.0 / self.scale_pos * torch.log(
        #        1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh)), dim=1) - pos_flase)
        neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh)), dim=1) - neg_flase)

        loss = pos_loss + neg_loss
        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)
        loss = torch.sum(pos_loss + neg_loss) / sim_mat.shape[0]
        # print(loss)
        pos_return = torch.exp(-self.scale_pos * (pos_pair - self.thresh))*pos_pair_tmp2.long() + abs(torch.min(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))*(-2) * neg_pair_tmp2.long()
        neg_return = torch.exp(self.scale_neg * (neg_pair - self.thresh))*neg_pair_tmp2.long() +abs(torch.min(torch.exp(self.scale_neg * (neg_pair - self.thresh))))*(-2) * pos_pair_tmp2.long()
        return loss, pos_return, neg_return