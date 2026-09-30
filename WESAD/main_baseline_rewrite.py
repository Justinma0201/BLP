import sys
import os
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
parser.add_argument('--base_path', type=str, default='./data/5fold_normalized')
parser.add_argument('--word_emb_path', type=str, default='./data/wesad_baseline_only_word_rewrite_embedding.csv')
parser.add_argument('--mask', type=int, default=1)
parser.add_argument('--mode', type=str, default='base')
parser.add_argument('--shuffle', type=int, default=1)
parser.add_argument('--loss_weight', type=int, default=1)
parser.add_argument('--lambda_inter', type=float, default=0.0, help='Weight for Inter-ID contrastive loss')
parser.add_argument('--lambda_intra', type=float, default=0.5, help='Weight for Intra-ID contrastive loss')
parser.add_argument('--lambda_align', type=float, default=0.0, help='Weight for Alignment loss')
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
parser.add_argument('--batch_size', type=int, default=16, help='Number of sequence chunks per batch')
parser.add_argument('--seq_len', type=int, default=128)
parser.add_argument('--window_stride_sec', type=float, default=0.25)
parser.add_argument('--contrastive_max_windows', type=int, default=512)
parser.add_argument('--accum_steps', type=int, default=1)
parser.add_argument('--encoder_layers', type=int, default=1)
parser.add_argument('--weight', type=int, default=499)
parser.add_argument('--class_loss', type=float, default=0.33)
parser.add_argument('--scale_pos', type=float, default=7)
parser.add_argument('--scale_neg', type=float, default=6)
parser.add_argument('--earlystop', type=int, default=0)
parser.add_argument('--earlystop_limit', type=int, default=40)
parser.add_argument('--run', type=str, default='run0')
parser.add_argument('--prediction_output_dir', type=str, default='./result_analysis',
                    help='Directory used to save per-record test predictions')
args = parser.parse_args()
if not 1 <= args.seq_len <= 5000:
    parser.error('--seq_len must be between 1 and 5000')
if args.batch_size < 1 or args.contrastive_max_windows < 1 or args.window_stride_sec <= 0 or args.accum_steps < 1:
    parser.error('batch size, contrastive cap, window stride and accum_steps must be positive')
print('Using contiguous chunks; --mask is retained for CLI compatibility but no longer samples windows.')
run = ["wesad_baseline_rewrite"]
num_heads = args.num_heads
num_encoder_layers = args.encoder_layers
word_emb_dim = 1024
original_d_model = 61
fusion_d_model = 256
d_model = original_d_model + word_emb_dim
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
    """Sinusoidal positional encoding for non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

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
        required_len = emb.size(0) if step is None else int(step) + 1
        if required_len > self.pe.size(0):
            position = torch.arange(required_len, device=self.pe.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.dim, 2, device=self.pe.device,
                                             dtype=torch.float) * (-math.log(10000.0) / self.dim))
            pe = torch.zeros(required_len, self.dim, device=self.pe.device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.pe = pe.unsqueeze(1)
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
        hrv_dim=61,
        embedding_dim=1024,
        fusion_dim=fusion_d_model,
    ):
        super(Transformer, self).__init__()
        self.concat_dim = fusion_dim * 2
        self.embedding_dim = embedding_dim
        self.hrv_dim = hrv_dim
        self.project = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 512),
            nn.GELU(),
            nn.Linear(512, fusion_dim)
        )

        self.hrv_project = nn.Sequential(
            nn.Linear(hrv_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

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
                    f"Found {all_padded.sum().item()} samples with all padding，"
                )
                
        hrv_feature = x[:, :, :self.hrv_dim]
        text_embedding_all = x[:, :, self.hrv_dim:]
        if input_mode == 'hrv_only':
            text_embedding_all = torch.zeros_like(text_embedding_all)
        elif input_mode == 'text_only':
            hrv_feature = torch.zeros_like(hrv_feature)

        text_proj = self.project(text_embedding_all)
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

        return class_output, feature, class_output_embedding, domain_output, encoded_hrv, text_proj, z_hrv, z_text, hrv_to_text_weight, text_to_hrv_weight


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_wesad_frame(frame, source):
    frame = frame.copy()
    if 'label_binary' not in frame and 'label_binarized' in frame:
        frame = frame.rename(columns={'label_binarized': 'label_binary'})
    features = [f'feature_{j}' for j in range(original_d_model)]
    required = ['ID', 'window_start_sec', 'label_binary'] + features
    missing = [col for col in required if col not in frame]
    if missing:
        raise ValueError(f"{source}: missing WESAD columns {missing}")
    if frame.empty or frame['ID'].isna().any():
        raise ValueError(f"{source}: data cannot be empty, ID cannot be missing")
    frame['ID'] = frame['ID'].astype(str).str.strip()
    if frame['ID'].eq('').any():
        raise ValueError(f"{source}: ID cannot be empty string")
    boundaries = [c for c in ('session_id', 'recording_id', 'segment_id') if c in frame]
    frame = frame[required + boundaries].copy()
    numeric = ['window_start_sec', 'label_binary'] + features
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"{source}: HRV、time or label_binary contains NaN/Inf, please handle first")
    if not frame['label_binary'].isin([0, 1]).all():
        raise ValueError(f"{source}: label_binary must be 0 or 1")
    frame['label_binary'] = frame['label_binary'].astype(int)
    if frame.duplicated(['ID', 'window_start_sec']).any():
        raise ValueError(f"{source}: duplicate ID + window_start_sec found")
    frame['record_id'] = frame['ID'] + '_' + frame['window_start_sec'].astype(str)
    return frame.sort_values(['ID', 'window_start_sec'], kind='stable').reset_index(drop=True)


class WindowChunkLoader:
    def __init__(self, frame, batch_size, shuffle=False):
        cols = ([f'feature_{j}' for j in range(original_d_model)]
                + [f'embedding_{j}' for j in range(word_emb_dim)])
        required = cols + ['ID', 'record_id', 'window_start_sec', 'label_binary']
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f'Missing columns: {missing}')
        if frame.columns.duplicated().any():
            raise ValueError('Duplicate dataframe columns')
        if frame.empty or frame['record_id'].duplicated().any():
            raise ValueError('Expected nonempty data with unique record_id per window')
        frame = frame.copy()
        frame['window_start_sec'] = pd.to_numeric(frame['window_start_sec'], errors='raise')
        if not np.isfinite(frame['window_start_sec']).all():
            raise ValueError('window_start_sec must be finite original timestamps')
        frame = frame.sort_values(['ID', 'window_start_sec'], kind='stable').reset_index(drop=True)
        self.values = frame[cols].to_numpy(dtype=np.float32)
        if not np.isfinite(self.values).all():
            raise ValueError('Model features contain nonfinite values')
        self.labels = frame['label_binary'].to_numpy(dtype=np.int64)
        if not np.isin(self.labels, [0, 1]).all():
            raise ValueError('label_binary must be 0 or 1')
        self.ids = frame['ID'].to_numpy()
        self.records = frame['record_id'].astype(str).to_numpy()
        self.pss = frame['pss'].to_numpy(dtype=np.float32) if 'pss' in frame else np.full(len(frame), -1, dtype=np.float32)
        self.batch_size, self.shuffle = batch_size, shuffle
        self.seq_len = args.seq_len
        self.chunks = []
        for _, group in frame.groupby('ID', sort=False):
            times = group['window_start_sec'].to_numpy()
            breaks = ~np.isclose(np.diff(times), args.window_stride_sec, rtol=0, atol=1e-6)
            for col in ('session_id', 'recording_id', 'segment_id'):
                if col in group:
                    values = group[col].to_numpy()
                    breaks |= values[1:] != values[:-1]
            boundaries = np.r_[0, np.flatnonzero(breaks) + 1, len(group)]
            offset = int(group.index[0])
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                for pos in range(int(left), int(right), self.seq_len):
                    self.chunks.append((offset + pos, offset + min(pos + self.seq_len, int(right))))
        print(f'Chunk loader: {len(frame)} windows, {len(self.chunks)} chunks, '
              f'batch_size={batch_size}, seq_len={self.seq_len}, shuffle={shuffle}')

    def __len__(self):
        return math.ceil(len(self.chunks) / self.batch_size)

    def __iter__(self):
        order = np.arange(len(self.chunks))
        if self.shuffle:
            np.random.shuffle(order)
        for pos in range(0, len(order), self.batch_size):
            chunks = [self.chunks[i] for i in order[pos:pos + self.batch_size]]
            shape = (len(chunks), self.seq_len)
            x = np.zeros((*shape, d_model), dtype=np.float32)
            y = np.full(shape, -1, dtype=np.int64)
            padding = np.ones(shape, dtype=bool)
            pss = np.full(shape, -1, dtype=np.float32)
            ids, records = [], []
            for i, (start, end) in enumerate(chunks):
                n = end - start
                x[i, :n] = self.values[start:end]
                y[i, :n] = self.labels[start:end]
                pss[i, :n] = self.pss[start:end]
                padding[i, :n] = False
                ids.extend(self.ids[start:end])
                records.extend(self.records[start:end])
            yield [torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(padding),
                   np.asarray(ids), torch.from_numpy(pss), np.asarray(records)]


def dataloader(x, mask_tf, batch_size, shuffle=False):
    return WindowChunkLoader(x, batch_size, shuffle=shuffle)


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
for run_name in run:
    seed_init(args.seed)
    wandb.init(project="ICASSP-2027", config=vars(args), reinit=True)
    wandb.run.name = f"baseline_rewrite_wesad"
    n_splits = 5
    #PATH = f'./model/model_{args.run}.pt'
    num_epoch = args.epochs

    train_total_loss = []
    val_total_loss = []
    y_pred_total = []
    y_test_total = []

    pre_uar = None
    uar_idx = 0

    word_emb_cols = [f"embedding_{i}" for i in range(word_emb_dim)]
    fold_list = ['fold-1', 'fold-2', 'fold-3', 'fold-4', 'fold-5']
    BASE_PATH = args.base_path
    for fold_name in fold_list:
        current_fold = int(fold_name.split('-')[1])
        print(f"\n================ {fold_name} ================")
        PATH = f'./model/model_{run_name}_{fold_name}.pt'

        word_emb_path = args.word_emb_path
        print(f"Loading embedding: {word_emb_path}")

        df_word_emb = pd.read_csv(word_emb_path, dtype={'record_id': str, 'user_id': str, 'ID': str})
        if not all(col in df_word_emb.columns for col in word_emb_cols):
            short_cols = [f"emb_{i}" for i in range(word_emb_dim)]
            if not all(col in df_word_emb.columns for col in short_cols):
                raise ValueError(f"{word_emb_path}: missing complete {word_emb_dim}-dimensional embedding columns")
            df_word_emb = df_word_emb.rename(columns=dict(zip(short_cols, word_emb_cols)))
        id_column = next((c for c in ['user_id', 'ID', 'record_id'] if c in df_word_emb), None)
        if id_column is None:
            raise ValueError(f"{word_emb_path}: missing user_id / ID / record_id")
        df_word_emb = df_word_emb[[id_column] + word_emb_cols].rename(columns={id_column: 'ID'})
        df_word_emb['ID'] = df_word_emb['ID'].str.strip()
        df_word_emb[word_emb_cols] = df_word_emb[word_emb_cols].apply(pd.to_numeric, errors='raise')
        if not np.isfinite(df_word_emb[word_emb_cols].to_numpy(dtype=float)).all():
            raise ValueError(f"{word_emb_path}: embedding contains NaN/Inf")
        if df_word_emb['ID'].isna().any() or df_word_emb['ID'].eq('').any() or df_word_emb['ID'].duplicated().any():
            raise ValueError(f"{word_emb_path}: subject ID cannot be missing or duplicated")

        print(f"Loaded {fold_name} embedding, samples = {len(df_word_emb)}, dim = {len(word_emb_cols)}")

        train_csv_path = os.path.join(BASE_PATH, fold_name, 'train_normalized.csv')
        test_csv_path = os.path.join(BASE_PATH, fold_name, 'test_normalized.csv')

        df_train = pd.read_csv(train_csv_path, dtype={'ID': str})
        df_test = pd.read_csv(test_csv_path, dtype={'ID': str})

        df_train = prepare_wesad_frame(df_train, f"{fold_name} train")
        df_test = prepare_wesad_frame(df_test, f"{fold_name} test")
        print(f'Number of train samples: {len(df_train)}')
        print(f'Number of test samples: {len(df_test)}')

        aligned_frames = []
        for split_name, frame in [('train', df_train), ('test', df_test)]:
            aligned = frame.merge(
                df_word_emb, on='ID', how='left', validate='many_to_one',
                sort=False, indicator=True,
            )
            missing_ids = aligned.loc[aligned['_merge'] != 'both', 'ID'].unique()
            if len(missing_ids):
                raise ValueError(f"{fold_name} {split_name}: missing subject ID in embedding: {missing_ids[:10].tolist()}")
            if aligned[word_emb_cols].isna().any().any():
                raise ValueError(f"{fold_name} {split_name}: embedding contains missing values")
            aligned_frames.append(aligned.drop(columns='_merge'))
        df_train, df_test = aligned_frames
        print(f"Training data label:\n{df_train['label_binary'].value_counts()}")

        if args.loss_weight == 1:
            if df_train['label_binary'].nunique() != 2:
                raise ValueError(f'{fold_name}: calculating class weights requires the training set to contain both 0 and 1 labels')
            class_weights = [1 / (len(df_train['label_binary']) - sum(df_train['label_binary'])), 1 / sum(df_train['label_binary'])]
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        if args.earlystop == 1:
            df_train_f = pd.DataFrame([])
            df_val_f = pd.DataFrame([])
            df_train_tmp = df_train.copy()

            df_unique_tmp = df_train_tmp[['ID', 'label_binary']].drop_duplicates(subset=['ID']).reset_index(drop=True)
            df_unique_tmp, df_val_tmp = train_test_split(df_unique_tmp, test_size=0.2, random_state=2)

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
            df_train = dataloader(df_train_f, 0, batch_size, shuffle=bool(args.shuffle))
            df_val = dataloader(df_val_f, 0, batch_size)
        else:
            df_train = dataloader(df_train, 0, batch_size, shuffle=bool(args.shuffle))
            df_val = df_test



        model = Transformer(original_d_model, num_heads, num_encoder_layers, dropout, hidden_channels, out_channels,
                            num_domain=num_domain, hrv_dim=original_d_model, embedding_dim=word_emb_dim, fusion_dim=fusion_d_model)
        model = model.to(device)

        ema = EMA(model, 0.9999)
        ema.register()

        class_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        domain_criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
        if args.loss_weight == 1:
            class_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        ms_loss_criterion = losses.MultiSimilarityLoss(alpha=args.scale_pos, beta=args.scale_neg).to(device)
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


                y_pred, embedding, stress_embedding, domain_output, pure_hrv_feature, text_proj, z_hrv, z_text, hrv_to_text_weight, text_to_hrv_weight = model(x=x.float(), alpha=alpha, mask=padding_mask, input_mode=args.input_mode)

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
                weighted_ms_loss = class_loss.new_zeros(())
                if len(y_valid) and any(w != 0 for w in (args.lambda_inter, args.lambda_intra, args.lambda_align)):
                    selected = torch.randperm(len(y_valid), device=device)[:args.contrastive_max_windows]
                    paired_hrv = z_hrv_valid[selected]
                    paired_text = z_text_valid[selected]
                    selected_y = y_valid[selected].long()
                    combined_features = torch.cat([paired_hrv, paired_text], dim=0)
                    if args.lambda_inter != 0:
                        labels = torch.cat([selected_y, selected_y])
                        weighted_ms_loss = weighted_ms_loss + args.lambda_inter * ms_loss_criterion(combined_features, labels)
                    if args.lambda_intra != 0:
                        selected_ids = ID_array[selected.cpu().numpy()]
                        keys = [f'{uid}_{int(lbl)}' for uid, lbl in zip(selected_ids, selected_y.cpu().numpy())]
                        _, codes = np.unique(keys, return_inverse=True)
                        labels = torch.as_tensor(codes, device=device, dtype=torch.long)
                        weighted_ms_loss = weighted_ms_loss + args.lambda_intra * ms_loss_criterion(combined_features, torch.cat([labels, labels]))
                    if args.lambda_align != 0:
                        labels = torch.arange(len(selected), device=device)
                        weighted_ms_loss = weighted_ms_loss + args.lambda_align * ms_loss_criterion(combined_features, torch.cat([labels, labels]))

                domain_loss = class_loss.new_zeros(())
                if args.lambda_domain != 0 and len(ID_array):
                    domain_output_flat = domain_output.reshape(-1, domain_output.shape[-1])
                    labels = torch.tensor([id_to_label[uid] for uid in ID_array], dtype=torch.long, device=device)
                    domain_loss = domain_criterion(domain_output_flat[valid_mask], labels)

                loss = class_loss + weighted_ms_loss + (args.lambda_domain * domain_loss)
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

                    y_pred, embedding, stress_embedding, domain_output, pure_hrv_feature, text_proj, z_hrv, z_text, hrv_to_text_weight, text_to_hrv_weight = model(x=x.float(), alpha=alpha, mask=padding_mask, input_mode=args.input_mode)
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

                wandb.log({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_uar": val_uar,
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
        record_ids_test = []
        alpha = 0
        

        with torch.no_grad():
            for i, testbatch in enumerate(df_test):
                p = float(i) / len(df_test)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                x = torch.Tensor(testbatch[0]).to(device)
                y = torch.Tensor(testbatch[1]).to(device)
                padding_mask = testbatch[2].to(device)
                batch_IDs = testbatch[3]
                batch_records = testbatch[5]

                y_pred, embedding, stress_embedding, domain_output, pure_hrv_feature, text_proj, z_hrv, z_text, hrv_to_text_weight, text_to_hrv_weight = model(x=x.float(), alpha=alpha, mask=padding_mask, input_mode=args.input_mode)

                y_ = []
                yp_ = []
                current_id_idx = 0
                for yl in range(len(y)):
                    tag = y.shape[1] - sum(padding_mask[yl])
                    tag = tag.data.cpu().numpy()
                    y_.extend(y[yl][:tag].tolist())
                    yp_.extend(y_pred[yl][:tag].tolist())
                    ID_test_.extend(batch_IDs[current_id_idx : current_id_idx + tag].tolist())
                    record_ids_test.extend(batch_records[current_id_idx : current_id_idx + tag].tolist())
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
        if len(record_ids_test) != len(df_test.records) or set(record_ids_test) != set(df_test.records):
            raise RuntimeError('Test windows were omitted or duplicated during chunk prediction')
        fold_df = pd.DataFrame({
            'ID': ID_test_,
            'record_id': record_ids_test,
            'y_true': y_test_,
            'y_prob': y_prob_,
            'y_pred': y_pred_
        })
        prediction_df = pd.DataFrame({
            'fold': fold_name,
            'ID': ID_test_,
            'record_id': record_ids_test,
            'gt': y_test_.astype(int),
            'pred': y_pred_.astype(int),
            'prob_0': 1.0 - np.asarray(y_prob_),
            'prob_1': np.asarray(y_prob_),
        })

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

        print(f"\nSaving {fold_name}'s metrics report")

        os.makedirs(args.prediction_output_dir, exist_ok=True)
        raw_output_path = os.path.join(
            args.prediction_output_dir,
            f"raw_predictions_baseline_rewrite_wesad_{args.input_mode}_{fold_name}.csv"
        )
        prediction_df.to_csv(raw_output_path, index=False)
        print(f"[{fold_name}] Raw predictions saved to: {raw_output_path}")

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
        print(f"\n[{fold_name}] Per-ID metrics:")
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
print("5-Fold Cross Validation result：")
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
