import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
import pdb
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import pandas as pd
import math
import numpy as np
import argparse
import wandb
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import os
from sklearn.preprocessing import StandardScaler
from random import sample
import random
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import matthews_corrcoef
import csv
import time
from multisimilarityloss import MultiSimilarityLoss
from pytorch_metric_learning import losses
import torch.nn.functional as F
from hist import *
import torch.optim as optim
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
os.makedirs('./model', exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument('--input_mode', type=str, default='full',
                     choices=['full', 'hrv_only', 'text_only'],
                     help='full: HRV+word, hrv_only: only HRV, text_only: only word')
parser.add_argument('--mask', type=int, default=1)
parser.add_argument('--mode', type=str, default='base')
parser.add_argument('--shuffle', type=int, default=1)
parser.add_argument('--loss_weight', type=int, default=1)
parser.add_argument('--lambda_inter', type=float, default=0.0, help='Weight for Inter-ID contrastive loss')
parser.add_argument('--lambda_intra', type=float, default=0.5, help='Weight for Intra-ID contrastive loss')
parser.add_argument('--lambda_align', type=float, default=0.0, help='Weight for Alignment loss')
parser.add_argument('--lambda_ortho', type=float, default=1.0, help='Weight for Orthogonal loss')
parser.add_argument('--lambda_domain', type=float, default=0.0, help='Weight for Domain (GRL) loss')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--select_fold', type=int, default=6)
parser.add_argument('--posenc', type=int, default=1)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--lr-ds', default=1e-1, type=float)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--num_heads', type=int, default=2)
parser.add_argument('--tau', default=32, type=float)
parser.add_argument('--alpha', default=0.9, type=float)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--accum_steps', type=int, default=1)
parser.add_argument('--encoder_layers', type=int, default=1)
parser.add_argument('--weight', type=int, default=499)
parser.add_argument('--class_loss', type=float, default=0.33)
parser.add_argument('--scale_pos', type=float, default=7)
parser.add_argument('--scale_neg', type=float, default=6)
parser.add_argument('--earlystop', type=int, default=0)
parser.add_argument('--earlystop_limit', type=int, default=40)
parser.add_argument('--run', type=str, default='run0')
args = parser.parse_args()
run = ["main"
       ]
num_heads = args.num_heads
num_encoder_layers = args.encoder_layers
word_emb_dim = 1024
num_embeddings = 4
original_d_model = 100
fusion_d_model = 100
d_model = original_d_model + word_emb_dim * num_embeddings
dropout = 0.3
hidden_channels = 16
out_channels = 2
batch_size = args.batch_size

def seed_init(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

from torch.autograd import Function

class PositionalEncoding(nn.Module):

    def __init__(self, dim, max_len=5000, dropout=0):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        return emb

class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
    
class OrthogonalLoss(nn.Module):
    def __init__(self):
        super(OrthogonalLoss, self).__init__()

    def forward(self, h1, h2):
        h1_norm = F.normalize(h1, p=2, dim=-1)
        h2_norm = F.normalize(h2, p=2, dim=-1)

        correlation_matrix = torch.matmul(h1_norm.t(), h2_norm)

        loss = torch.sum(correlation_matrix ** 2) / (h1.size(0) ** 2)
        return loss


class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        num_encoder_layers,
        dropout,
        hidden_channels,
        out_channels,
        num_domain,
        hrv_dim=100,
        embedding_dim=1024,
        fusion_dim=fusion_d_model,
        num_embeddings=4,
    ):
        super(Transformer, self).__init__()
        self.project = nn.Linear(embedding_dim, fusion_dim)
        self.hrv_project = nn.Linear(hrv_dim, fusion_dim)
        self.concat_dim = fusion_dim * 2
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.source_projections = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) for _ in range(num_embeddings)
        ])
        self.fusion_weights = nn.Parameter(torch.ones(self.num_embeddings))

        self.proj_head_hrv = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, fusion_dim)
        )
        self.proj_head_text = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, fusion_dim)
        )

        self.pos_encoder = PositionalEncoding(dim=fusion_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=fusion_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.hrv_to_text_attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.text_to_hrv_attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        self.norm_hrv = nn.LayerNorm(fusion_dim)
        self.norm_text = nn.LayerNorm(fusion_dim)

        self.class_classifier1 = nn.Sequential()
        self.class_classifier2 = nn.Sequential()
        self.class_classifier1.add_module('c_fc1', nn.Linear(self.concat_dim, hidden_channels))
        self.class_classifier2.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier2.add_module('c_dropout1', nn.Dropout(dropout))
        self.class_classifier2.add_module('c_fc2', nn.Linear(hidden_channels, out_channels))

        self.domain_classifier1 = nn.Sequential()
        self.domain_classifier2 = nn.Sequential()
        self.domain_classifier1.add_module('d_fc1', nn.Linear(self.concat_dim, hidden_channels))
        self.domain_classifier2.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier2.add_module('d_dropout1', nn.Dropout(dropout))
        self.domain_classifier2.add_module('d_fc2', nn.Linear(hidden_channels, num_domain))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, alpha, mask=None, input_mode='full'):
        self.mask = mask
        
        if self.mask is not None:
            all_padded = self.mask.all(dim=1)
            if all_padded.any():
                raise RuntimeError(
                    f"Found {all_padded.sum().item()} samples that are all padding,"
                )
                
        hrv_feature = x[:, :, :100]
        text_embedding_all = x[:, :, 100:]
        if input_mode == 'hrv_only':
            text_embedding_all = torch.zeros_like(text_embedding_all)
        elif input_mode == 'text_only':
            hrv_feature = torch.zeros_like(hrv_feature)
        B, S, _ = text_embedding_all.shape
        text_embedding_all = text_embedding_all.reshape(B, S, self.num_embeddings, self.embedding_dim)
        projected_sources = []
        for k in range(self.num_embeddings):
            proj = self.source_projections[k](text_embedding_all[:, :, k, :])
            projected_sources.append(proj)
            
        stacked_sources = torch.stack(projected_sources, dim=2)
        weights = F.softmax(self.fusion_weights, dim=0)
        fused_text_1024 = torch.einsum('bsne,n->bse', stacked_sources, weights)

        text_proj = self.project(fused_text_1024)
        hrv_proj = self.hrv_project(hrv_feature)

        x_new = hrv_proj

        if args.posenc == 1:
            x_new = x_new.permute(1, 0, 2)
            x_new = self.pos_encoder(x_new)
            x_new = x_new.permute(1, 0, 2)

        encoded_hrv = self.encoder(x_new, src_key_padding_mask=self.mask)

        z_hrv = self.proj_head_hrv(encoded_hrv)
        z_text = self.proj_head_text(text_proj)

        hrv_attended, hrv_to_text_weight = self.hrv_to_text_attn(
            query=encoded_hrv,
            key=text_proj,
            value=text_proj,
            key_padding_mask=self.mask 
        )
        hrv_fused = self.norm_hrv(encoded_hrv + hrv_attended)

        text_attended, text_to_hrv_weight = self.text_to_hrv_attn(
            query=text_proj,
            key=encoded_hrv,
            value=encoded_hrv,
            key_padding_mask=self.mask 
        )
        text_fused = self.norm_text(text_proj + text_attended)

        feature = torch.cat([hrv_fused, text_fused], dim=-1)

        class_output_embedding = self.class_classifier1(feature)
        class_output = self.class_classifier2(class_output_embedding)
        reversed_feature = GRL.apply(feature, alpha)
        domain_output_embedding = self.domain_classifier1(reversed_feature)
        domain_output = self.domain_classifier2(domain_output_embedding)

        return class_output, feature, class_output_embedding, domain_output, encoded_hrv, text_proj, z_hrv, z_text, projected_sources, hrv_to_text_weight, text_to_hrv_weight


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mask(x, r, cap=None):
    n_keep = max(1, round(len(x) * r))
    if cap is not None:
        n_keep = min(n_keep, cap)
    x = x.sample(n=n_keep)
    x = x.sort_values(by='date')
    return x


def dataloader(x, mask_tf, batch_size):
    subject = x['ID'].unique()
    subject = pd.DataFrame(subject)

    df = []
    num = math.ceil(len(subject) / batch_size)

    if args.shuffle == 1:
        group_name = x.ID.unique()
        np.random.shuffle(group_name)

    for n in range(num):
        start_idx = n * batch_size
        end_idx = min((n + 1) * batch_size, len(subject))
        batch_group_names = group_name[start_idx:end_idx]
        participant = x.groupby(x['ID'])
        batch = pd.concat([participant.get_group(group) for group in batch_group_names])
        max_value = max(batch['ID'].value_counts())
        if mask_tf == 1:
            lb = 0.2 
            r = random.uniform(lb, 1)
            max_value = max(1, round(max_value * r))
        else:
            r = 1.0

        padding_mask = np.zeros([len(subject.index[n * batch_size:(n + 1) * batch_size]), max_value])
        batch_name = batch.ID.unique()
        mask_num = 0

        batch_x_array = np.zeros([len(batch_name), max_value, d_model])
        batch_y_array = np.zeros([len(batch_name), max_value]) - 1
        batch_ID_array = np.array([])
        batch_pss_array = np.zeros([len(batch_name), max_value]) - 1

        for i in range(len(batch_name)):
            tmp = batch[lambda x: x.ID == batch_name[i]].copy()

            tmp = tmp.sort_values(by='date')

            if mask_tf == 1:
                tmp = mask(tmp, r, cap=max_value)

            add_value = max_value - len(tmp)

            feature_cols = [c for c in tmp.columns if c not in ['ID', 'date', 'record_id', 'label_binarized', 'pss', 'stress']]

            batch_x_array[i][:len(tmp)] = tmp[feature_cols].values
            batch_y_array[i][:len(tmp)] = tmp['label_binarized'].values
            batch_ID_array = np.concatenate((batch_ID_array, tmp['ID'].values))

            if 'pss' in tmp.columns:
                batch_pss_array[i][:len(tmp)] = tmp['pss'].values

            if add_value != 0:
                padding_mask[mask_num][-add_value:] = np.ones(add_value)
            mask_num = mask_num + 1

        batch_x_array = torch.tensor(batch_x_array, dtype=torch.float32)
        batch_y_array = torch.tensor(batch_y_array, dtype=torch.float32)
        src_key_padding_mask = torch.tensor(padding_mask, dtype=torch.bool)
        batch_pss_array = torch.tensor(batch_pss_array, dtype=torch.float32)
        df.append([batch_x_array, batch_y_array, src_key_padding_mask, batch_ID_array, batch_pss_array])

    return df


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
    def get_state_dict(self):
        return self.shadow

experiment_results = {r: {"acc": [], "sens": [], "spec": [], "uar": [], "f1": [], "mcc": []} for r in run}
roc_data = {r: {'y_true': [], 'y_prob': []} for r in run}
per_id_data = {r: pd.DataFrame(columns=['ID', 'y_true', 'y_prob', 'y_pred']) for r in run}
for run_name in run:
    seed_init(args.seed)
    wandb.init(project="ICASSP-2027", config=vars(args), reinit=True)
    wandb.run.name = f"proposed"
    n_splits = 5
    #PATH = f'./model/model_{args.run}.pt'
    num_epoch = args.epochs

    train_total_loss = []
    val_total_loss = []
    y_pred_total = []
    y_test_total = []

    pre_uar = None
    uar_idx = 0

    print("\nLoading global word embeddings (all samples)...")
    word_emb_paths = [
        "./data/hrv_time_full_embedding.csv",
        "./data/hrv_freq_full_embedding.csv",
        "./data/hrv_ms_rr_full_embedding.csv",
        "./data/hrv_ms_drr_full_embedding.csv",
    ]
    df_word_emb_list = []
    word_emb_cols_by_src = [] 

    for src_idx, path in enumerate(word_emb_paths):
        df_e = pd.read_csv(path)
        df_e['record_id'] = df_e['record_id'].astype(str)

        rename_map = {f"emb_{i}": f"emb_src{src_idx}_{i}" for i in range(word_emb_dim)}
        df_e = df_e.rename(columns=rename_map)
        cols = list(rename_map.values())
        word_emb_cols_by_src.append(cols)
        df_word_emb_list.append(df_e[['record_id'] + cols])

    word_emb_cols = [c for cols in word_emb_cols_by_src for c in cols]
    print(f"Loaded {len(word_emb_paths)} embedding sources, total dim = {len(word_emb_cols)}")
    fold_list = ['fold-1', 'fold-2', 'fold-3', 'fold-4', 'fold-5']
    BASE_PATH = './data/HRV_I5F_Tiles'
    for fold_name in fold_list:
        current_fold = int(fold_name.split('-')[1])
        print(f"\n================ {fold_name} ================")
        PATH = f'./model/model_{run_name}_{fold_name}.pt'

        train_csv_path = os.path.join(BASE_PATH, fold_name, 'df_train_etc_stats_selectnum100.csv')
        test_csv_path = os.path.join(BASE_PATH, fold_name, 'df_test_etc_stats_selectnum100.csv')

        df_train = pd.read_csv(train_csv_path)
        df_test = pd.read_csv(test_csv_path)

        df_train.dropna(inplace=True)
        df_test.dropna(inplace=True)
        df_train.drop_duplicates(inplace=True)
        df_test.drop_duplicates(inplace=True)

        print(f'Number of train samples: {len(df_train)}')
        print(f'Number of test samples: {len(df_test)}')

        columns_train = list(df_train.columns)
        columns_test = list(df_test.columns)

        df_train['record_id'] = df_train['ID'].astype(str) + '_' + df_train['date'].astype(str)
        df_test['record_id'] = df_test['ID'].astype(str) + '_' + df_test['date'].astype(str)

        for df_e in df_word_emb_list:
            df_train = pd.merge(df_train, df_e, on='record_id', how='inner')
            df_test = pd.merge(df_test, df_e, on='record_id', how='inner')
        print(f"Training data label:\n{df_train['label_binarized'].value_counts()}")

        if 'label_binarized' in columns_train:
            columns_train.pop(columns_train.index('label_binarized'))
        columns_train.extend(word_emb_cols)
        columns_train.append('label_binarized')
        df_train = df_train[columns_train]

        if 'label_binarized' in columns_test:
            columns_test.pop(columns_test.index('label_binarized'))
        columns_test.extend(word_emb_cols)
        columns_test.append('label_binarized')
        df_test = df_test[columns_test]

        df_train.dropna(inplace=True)
        df_test.dropna(inplace=True)

        if args.loss_weight == 1:
            class_weights = [1 / (len(df_train['label_binarized']) - sum(df_train['label_binarized'])), 1 / sum(df_train['label_binarized'])]
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        if args.earlystop == 1:
            df_train_f = pd.DataFrame([])
            df_val_f = pd.DataFrame([])
            df_train_tmp = df_train.copy()

            df_unique_tmp = df_train_tmp[['ID', 'label_binarized']].drop_duplicates(subset=['ID']).reset_index(drop=True)
            df_unique_tmp, df_val_tmp = train_test_split(df_unique_tmp, test_size=0.2, stratify=df_unique_tmp['label_binarized'], random_state=2)

            for parti_idx in df_unique_tmp.ID:
                df_train_f = pd.concat([df_train_f, df_train_tmp[df_train_tmp.ID == parti_idx]])
            for parti_idx in df_val_tmp.ID:
                df_val_f = pd.concat([df_val_f, df_train_tmp[df_train_tmp.ID == parti_idx]])
            df_train_f.reset_index(drop=True, inplace=True)
            df_val_f.reset_index(drop=True, inplace=True)
        df_test = dataloader(df_test, 0, batch_size)
        if args.earlystop == 1:
            unique_train_ids = df_train_f['ID'].unique()
        else:
            unique_train_ids = df_train['ID'].unique()
            
        num_domain = len(unique_train_ids)

        id_to_label = {uid: idx for idx, uid in enumerate(unique_train_ids)}
        
        if args.earlystop == 1:
            df_train = dataloader(df_train_f, args.mask, batch_size)
            df_val = dataloader(df_val_f, 0, batch_size)
        else:
            df_train = dataloader(df_train, args.mask, batch_size)
            df_val = df_test



        model = Transformer(original_d_model, num_heads, num_encoder_layers, dropout, hidden_channels, out_channels,
                            num_domain=num_domain, hrv_dim=original_d_model, embedding_dim=word_emb_dim, fusion_dim=fusion_d_model, num_embeddings=num_embeddings)
        model = model.to(device)

        ema = EMA(model, 0.9999)
        ema.register()

        class_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        domain_criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
        if args.loss_weight == 1:
            class_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        ms_loss_criterion = losses.MultiSimilarityLoss(alpha=args.scale_pos, beta=args.scale_neg).to(device)
        ortho_criterion = OrthogonalLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for p in model.parameters():
            p.requires_grad = True

        epoch_loss = []
        epoch_acc = []
        test_epoch_loss = []
        test_epoch_acc = []

        print('Fold-{}'.format(int(fold_name.split('-')[1])))
        print(f'mask{args.mask}_posenc{args.posenc}_epoch{num_epoch}_lr{args.lr}_nheads{args.num_heads}')

        pre_uar = None
        uar_idx = 0
        train_pos_array = []
        train_neg_array = []
        test_pos_array = []
        test_neg_array = []

        for epoch in range(num_epoch):
            start_time = time.time()
            total_loss = 0
            model.train()

            y_pred_ = []
            y_test_ = []
            yp_a = []
            embedding_all = []
            stress_embedding_total = []
            stress_embedding_train = []
            stress_embedding_test = []
            ID_train = []
            total_batches = len(df_train)

            for i, trainbatch in enumerate(df_train):
                p = float(i + epoch * len(df_train)) / num_epoch / len(df_train)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                if i % args.accum_steps == 0:
                    optimizer.zero_grad()
                    group_start = i
                    group_end = min(i + args.accum_steps, total_batches)
                    current_accum = group_end - group_start
                x = torch.Tensor(trainbatch[0]).to(device)
                y = torch.Tensor(trainbatch[1]).to(device)
                padding_mask = trainbatch[2].to(device)
                ID = trainbatch[3]
                pss = torch.Tensor(trainbatch[4]).to(device)


                y_pred, embedding, stress_embedding, domain_output, pure_hrv_feature, text_proj, z_hrv, z_text, projected_sources, hrv_to_text_weight, text_to_hrv_weight = model(x=x.float(), alpha=alpha, mask=padding_mask, input_mode=args.input_mode)

                y_ = []
                yp_ = []
                embedding_ = torch.tensor([]).to(device)
                label_ = torch.tensor([]).to(device)

                for yl in range(len(y)):
                    tag = y.shape[1] - sum(padding_mask[yl])
                    tag = tag.data.cpu().numpy()

                    y_.extend(y[yl][:tag].tolist())
                    yp_.extend(y_pred[yl][:tag].tolist())
                    embedding_ = torch.cat((embedding_, embedding[yl][:tag]))
                    label_ = torch.cat((label_, y[yl][:tag]))
                    stress_embedding_total.extend(stress_embedding[yl][:tag].tolist())
                    stress_embedding_train.extend(stress_embedding[yl][:tag].tolist())

                ID_train.extend(ID)
                y_flat = y.flatten()
                y_pred_flat = y_pred.reshape(y_pred.shape[0] * y_pred.shape[1], -1)

                z_hrv_flat = z_hrv.reshape(-1, z_hrv.shape[-1])
                z_text_flat = z_text.reshape(-1, z_text.shape[-1])

                class_loss = class_criterion(y_pred_flat, y_flat.to(torch.int64))

                valid_mask = (y_flat != -1)
                y_valid = y_flat[valid_mask]
                z_hrv_valid = z_hrv_flat[valid_mask]
                z_text_valid = z_text_flat[valid_mask]

                ID_array = np.array(ID)
                y_valid_array = y_valid.cpu().numpy()

                intra_classes = [f"{uid}_{int(lbl)}" for uid, lbl in zip(ID_array, y_valid_array)]

                _, intra_labels_np = np.unique(intra_classes, return_inverse=True)
                intra_labels = torch.tensor(intra_labels_np).to(device)

                if len(y_valid) > 0:
                    combined_features = torch.cat([z_hrv_valid, z_text_valid], dim=0)

                    combined_inter_labels = torch.cat([y_valid, y_valid], dim=0)
                    ms_loss_inter = ms_loss_criterion(combined_features, combined_inter_labels.to(torch.int64))

                    combined_intra_labels = torch.cat([intra_labels, intra_labels], dim=0)
                    ms_loss_intra = ms_loss_criterion(combined_features, combined_intra_labels.to(torch.int64))
                    
                    N_valid = z_hrv_valid.size(0)
                    day_labels = torch.arange(N_valid, device=device)
                    combined_day_labels = torch.cat([day_labels, day_labels], dim=0)
                    ms_loss_align = ms_loss_criterion(combined_features, combined_day_labels)
                    weighted_ms_loss = (args.lambda_inter * ms_loss_inter) + \
                                       (args.lambda_intra * ms_loss_intra) + \
                                       (args.lambda_align * ms_loss_align)
                else:
                    weighted_ms_loss = torch.tensor(0.0).to(device)
                
                ortho_loss = torch.tensor(0.0).to(device)
                domain_output_flat = domain_output.reshape(domain_output.shape[0] * domain_output.shape[1], -1)
                valid_domain_labels = [id_to_label[uid] for uid in ID_array]
                valid_domain_labels_tensor = torch.tensor(valid_domain_labels, dtype=torch.int64).to(device)
                valid_domain_preds = domain_output_flat[valid_mask]
                
                if len(valid_domain_labels) > 0:
                    domain_loss = domain_criterion(valid_domain_preds, valid_domain_labels_tensor)
                else:
                    domain_loss = torch.tensor(0.0).to(device)
                if len(y_valid) > 0:
                    valid_projs = []
                    for proj in projected_sources:
                        proj_flat = proj.reshape(-1, proj.shape[-1])
                        valid_projs.append(proj_flat[valid_mask])
                    for idx1 in range(len(valid_projs)):
                        for idx2 in range(idx1 + 1, len(valid_projs)):
                            ortho_loss += ortho_criterion(valid_projs[idx1], valid_projs[idx2])
                loss = class_loss + weighted_ms_loss + (args.lambda_ortho * ortho_loss) + (args.lambda_domain * domain_loss)
                loss = loss / current_accum

                loss.backward()
                if (i + 1) % args.accum_steps == 0 or (i + 1) == len(df_train):
                    optimizer.step()
                    ema.update()

                total_loss += loss.item() * current_accum

                embedding_ = embedding_.data.cpu().numpy()
                y_pred = yp_
                yp_ = np.array(yp_)
                y_pred = np.argmax(y_pred, axis=1)
                y_test = y_

                y_pred_.append(y_pred)
                y_test_.append(y_test)
                yp_a.append(yp_)
                embedding_all.append(embedding_)
                num = i

            yp_a = np.concatenate(yp_a)
            embedding_all = np.concatenate(embedding_all)
            y_pred_ = np.concatenate(y_pred_)
            y_test_ = np.concatenate(y_test_)

            cm = confusion_matrix(y_test_, y_pred_)
            epoch_loss.append(total_loss / (num + 1))
            train_loss = total_loss / (num + 1)
            train_acc = accuracy_score(y_test_, y_pred_)
            epoch_acc.append(train_acc)

            ema.apply_shadow()
            model.eval()

            total_loss = 0
            y_pred_ = []
            y_test_ = []
            embedding_all = []
            yp_a = []
            alpha = 0
            ID_test = []
            sample_hrv2text_w = None
            sample_text2hrv_w = None

            with torch.no_grad():
                for i, valbatch in enumerate(df_val):
                    p = float(i) / len(df_val)
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                    x = torch.Tensor(valbatch[0]).to(device)
                    y = torch.Tensor(valbatch[1]).to(device)
                    padding_mask = valbatch[2].to(device)
                    ID = valbatch[3]
                    pss = torch.Tensor(valbatch[4]).to(device)

                    y_pred, embedding, stress_embedding, domain_output, pure_hrv_feature, text_proj, z_hrv, z_text, projected_sources, hrv_to_text_weight, text_to_hrv_weight = model(x=x.float(), alpha=alpha, mask=padding_mask, input_mode=args.input_mode)
                    if i == 0:
                        sample_hrv2text_w = hrv_to_text_weight[0].cpu().numpy()
                        sample_text2hrv_w = text_to_hrv_weight[0].cpu().numpy()
                    y_ = []
                    yp_ = []
                    embedding_ = torch.tensor([]).to(device)
                    label_ = torch.tensor([]).to(device)

                    for yl in range(len(y)):
                        tag = y.shape[1] - sum(padding_mask[yl])
                        tag = tag.data.cpu().numpy()

                        y_.extend(y[yl][:tag].tolist())
                        yp_.extend(y_pred[yl][:tag].tolist())
                        embedding_ = torch.cat((embedding_, embedding[yl][:tag]))
                        label_ = torch.cat((label_, y[yl][:tag]))
                        stress_embedding_total.extend(stress_embedding[yl][:tag].tolist())
                        stress_embedding_test.extend(stress_embedding[yl][:tag].tolist())

                    ID_test.extend(ID)
                    y = y.flatten()
                    y_pred = y_pred.reshape(y_pred.shape[0] * y_pred.shape[1], -1)

                    class_loss = class_criterion(y_pred, y.to(torch.int64))
                    loss = class_loss

                    total_loss += loss.item()
                    num = i

                    y_pred = yp_
                    yp_ = np.array(yp_)
                    y_pred = np.argmax(y_pred, axis=1)
                    y_test = y_

                    y_pred_.append(y_pred)
                    y_test_.append(y_test)
                    yp_a.append(yp_)
                    embedding_ = embedding_.data.cpu().numpy()
                    embedding_all.append(embedding_)

                yp_a = np.concatenate(yp_a)
                embedding_all = np.concatenate(embedding_all)
                y_pred_ = np.concatenate(y_pred_)
                y_test_ = np.concatenate(y_test_)

                test_acc = accuracy_score(y_test_, y_pred_)
                val_uar = recall_score(y_test_, y_pred_, average='macro')

                if args.earlystop == 1:
                    if pre_uar is None or val_uar > pre_uar:
                        torch.save({'model': model.state_dict(), 'ema': ema.get_state_dict()}, PATH)
                        pre_uar = val_uar
                        uar_idx = 0
                    else:
                        if epoch > 50: 
                            uar_idx += 1
                    
                    if uar_idx >= args.earlystop_limit:
                        print(f"Early stopping triggered at epoch {epoch}")
                        ema.restore()
                        break

                test_loss = total_loss / (num + 1)
                test_epoch_loss.append(test_loss)
                
                with torch.no_grad():
                    raw_weights = model.fusion_weights
                    normalized_weights = F.softmax(raw_weights, dim=0).cpu().numpy()
                    weight_names = ["weight_time", "weight_freq", "weight_ms_rr", "weight_ms_drr"]
                    weight_dict = {name: val for name, val in zip(weight_names, normalized_weights)}

                wandb.log({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_uar": val_uar,
                    **weight_dict
                })

            end_time = time.time()
            elapsed_time = end_time - start_time

            if (epoch + 1) % 10 == 0:
                print("epoch{}, Train acc: {:.3f}, Train loss: {:.3f}, test loss: {:.3f}, test acc: {:.3f}, val uar {:.3f}".format(
                    epoch + 1, train_acc, train_loss, test_loss, test_acc, val_uar))
            ema.restore()

        ema.apply_shadow()
        if args.earlystop == 1:
            checkpoint = torch.load(PATH)
            model.load_state_dict(checkpoint['model'])
            ema.shadow = checkpoint['ema']
            ema.apply_shadow()

        model.eval()
        y_pred_ = []
        y_test_ = []
        y_prob_ = []
        ID_test_ = []
        alpha = 0
        

        with torch.no_grad():
            for i, testbatch in enumerate(df_test):
                p = float(i) / len(df_test)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                x = torch.Tensor(testbatch[0]).to(device)
                y = torch.Tensor(testbatch[1]).to(device)
                padding_mask = testbatch[2].to(device)
                batch_IDs = testbatch[3]

                y_pred, embedding, stress_embedding, domain_output, pure_hrv_feature, text_proj, z_hrv, z_text, projected_sources, hrv_to_text_weight, text_to_hrv_weight = model(x=x.float(), alpha=alpha, mask=padding_mask, input_mode=args.input_mode)

                y_ = []
                yp_ = []
                current_id_idx = 0
                for yl in range(len(y)):
                    tag = y.shape[1] - sum(padding_mask[yl])
                    tag = tag.data.cpu().numpy()
                    y_.extend(y[yl][:tag].tolist())
                    yp_.extend(y_pred[yl][:tag].tolist())
                    ID_test_.extend(batch_IDs[current_id_idx : current_id_idx + tag].tolist())
                    current_id_idx += tag

                yp_tensor = torch.tensor(yp_)
                probs = F.softmax(yp_tensor, dim=1)[:, 1].numpy() 

                y_pred_hard = np.argmax(yp_, axis=1)

                y_pred_.extend(y_pred_hard)
                y_test_.extend(y_)
                y_prob_.extend(probs)

        ema.restore()
        y_pred_ = np.array(y_pred_)
        y_test_ = np.array(y_test_)
        roc_data[run_name]['y_true'].extend(y_test_)
        roc_data[run_name]['y_prob'].extend(y_prob_)
        fold_df = pd.DataFrame({
            'ID': ID_test_,
            'y_true': y_test_,
            'y_prob': y_prob_,
            'y_pred': y_pred_
        })
        per_id_data[run_name] = pd.concat([per_id_data[run_name], fold_df], ignore_index=True)

        cm = confusion_matrix(y_test_, y_pred_)

        fold_acc = accuracy_score(y_test_, y_pred_)
        fold_f1 = f1_score(y_test_, y_pred_)
        fold_uar = recall_score(y_test_, y_pred_, average='macro')
        fold_mcc = matthews_corrcoef(y_test_, y_pred_)
        fold_sens = recall_score(y_test_, y_pred_)
        fold_spec = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0.0

        experiment_results[run_name]["acc"].append(fold_acc)
        experiment_results[run_name]["sens"].append(fold_sens)
        experiment_results[run_name]["spec"].append(fold_spec)
        experiment_results[run_name]["uar"].append(fold_uar)
        experiment_results[run_name]["f1"].append(fold_f1)
        experiment_results[run_name]["mcc"].append(fold_mcc)

        print(f"[{run_name} - {fold_name}] UAR: {fold_uar:.3f}, ACC: {fold_acc:.3f}, F1: {fold_f1:.3f}", f"Sens: {fold_sens:.3f}, Spec: {fold_spec:.3f}, MCC: {fold_mcc:.3f}")

        wandb.log({
            "fold_test/accuracy": fold_acc,
            "fold_test/f1_score": fold_f1,
            "fold_test/uar": fold_uar,
            "fold_test/mcc": fold_mcc,
            "fold_test/sensitivity": fold_sens,
            "fold_test/specificity": fold_spec
        })

        print(f"\nSaving {fold_name} results and metrics...")

        raw_output_path = f"./result_analysis/raw_predictions_tiles_{fold_name}.csv"
        fold_df.to_csv(raw_output_path, index=False)
        print(f"[{fold_name}]'s raw predictions saved to: {raw_output_path}")

        id_metrics = []
        for subject_id, group in fold_df.groupby('ID'):
            y_t = group['y_true'].values.astype(int)
            y_p = group['y_pred'].values.astype(int)
            
            total_samples = len(y_t)
            pos_ratio = sum(y_t) / total_samples if total_samples > 0 else 0
            
            acc = accuracy_score(y_t, y_p)
            f1 = f1_score(y_t, y_p, zero_division=0)
            
            try:
                uar = recall_score(y_t, y_p, average='macro', zero_division=0)
            except:
                uar = np.nan
                
            try:
                mcc = matthews_corrcoef(y_t, y_p)
            except:
                mcc = np.nan
                
            id_metrics.append({
                'ID': subject_id,
                'Total_Samples': total_samples,
                'Actual_Stress_Ratio': round(pos_ratio, 3),
                'Accuracy': round(acc, 4),
                'F1_Score': round(f1, 4),
                'UAR': round(uar, 4),
                'MCC': round(mcc, 4)
            })
            
        df_metrics = pd.DataFrame(id_metrics)
        print(f"\n[{fold_name}] Per-ID Statistics:")
        print(df_metrics.to_string(index=False))
        print("\n")

    if len(experiment_results[run_name]['acc']) > 0:
        wandb.log({
            "Final_AVG_Accuracy": np.mean(experiment_results[run_name]["acc"]),
            "Final_AVG_F1": np.mean(experiment_results[run_name]["f1"]),
            "Final_AVG_UAR": np.mean(experiment_results[run_name]["uar"]),
            "Final_AVG_MCC": np.mean(experiment_results[run_name]["mcc"]),
            "Final_AVG_Sens": np.mean(experiment_results[run_name]["sens"]),
            "Final_AVG_Spec": np.mean(experiment_results[run_name]["spec"])
        })
    wandb.finish()
print("\n" + "="*60)
print("5-Fold Cross Validation results:")
print("="*60)

for run_name, metrics in experiment_results.items():
    if len(metrics['acc']) > 0:
        avg_acc = np.mean(metrics["acc"])
        avg_sens = np.mean(metrics["sens"])
        avg_spec = np.mean(metrics["spec"])
        avg_uar = np.mean(metrics["uar"])
        avg_f1 = np.mean(metrics["f1"])
        avg_mcc = np.mean(metrics["mcc"])
        std_uar = np.std(metrics["uar"])
        
        print(f"\nExperiment: {run_name}")
        print("-" * 30)
        print(f"  Avg Accuracy : {avg_acc:.4f}")
        print(f"  Avg Sens     : {avg_sens:.4f}")
        print(f"  Avg Spec     : {avg_spec:.4f}")
        print(f"  Avg UAR      : {avg_uar:.4f} (±{std_uar:.4f})")
        print(f"  Avg F1 Score : {avg_f1:.4f}")
        print(f"  Avg MCC      : {avg_mcc:.4f}")
        print("-" * 30)

print("Pipeline Successfully Finished.")