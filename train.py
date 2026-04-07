import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, f1_score
import warnings
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")
from collections import Counter
import random
import math 

# --- 1. 超参数与配置 (Hyperparameters and Configuration) ---
class Config:
    # 文件路径 (请根据实际情况修改)
    BASE_PATH = '/media/4T2/lmd/Qwen2.5/drug/Davis/final'
    DRUGBANK_FILE = os.path.join(BASE_PATH, 'Davis.txt')
    DRUG_MAP_FILE = os.path.join(BASE_PATH, 'drug_mapping.csv')
    PROTEIN_MAP_FILE = os.path.join(BASE_PATH, 'protein_mapping.csv')
    DRUG_FEATURE_FILE = os.path.join(BASE_PATH, 'drug_merge_1024.npy')
    PROTEIN_FEATURE_FILE = os.path.join(BASE_PATH, 'protein_merge_1024.npy')
    NUM_WORKERS = 12
    
    # 模型保存与日志
    MODEL_SAVE_PATH = '/media/4T2/lmd/Qwen2.5/drug/cold_zc/ours_all/warm/davis_best_model.pth'
    LOG_FILE = '/media/4T2/lmd/Qwen2.5/drug/cold_zc/ours_all/warm/davis_training_log.txt'

    # 模型参数
    EMBED_DIM = 1024
    ATTN_HEADS = 8 
    DROPOUT = 0.1

    # 训练参数
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 200
    EARLY_STOPPING_PATIENCE = 30


# --- 2. 模型架构定义 (Model Architecture) ---

# [新增] 正弦位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        :param d_model: 特征维度 (这里是 1024)
        :param dropout: Dropout 概率
        :param max_len: 预计算的最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个 (max_len, d_model) 的矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数下标用 sin, 奇数下标用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加 batch 维度: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer，这样它会被保存到 state_dict 中，但不会被 optimizer 更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: 输入张量 (Batch, Seq_len, d_model)
        """
        # 将位置编码加到输入 x 上 (截取 x 的当前长度)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LightweightAttention(nn.Module):
    """
    轻量级自注意力机制，用于聚合序列信息
    """
    def __init__(self, input_dim):
        super(LightweightAttention, self).__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x, mask):
        # x: (Batch, Seq_len, Dim)
        # mask: (Batch, Seq_len) True for padding
        attention_scores = self.attention_layer(x).squeeze(-1)
        attention_scores.masked_fill_(mask, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
        context_vector = torch.sum(x * attention_weights, dim=1)
        return context_vector


class DPIPredictor(nn.Module):
    def __init__(self, embed_dim, nhead, dropout=0.1):
        super(DPIPredictor, self).__init__()
        self.embed_dim = embed_dim  # 1024
        self.semantic_dim = embed_dim // 2 
        self.structural_dim = embed_dim // 2 
        
        # [修改点 1] 初始化位置编码层
        # 如果你想用可学习的 embedding，可以替换为: nn.Embedding(max_len, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)

        self.dropout_layer = nn.Dropout(dropout)
        
        # LayerNorm
        self.norm_drug_sem = nn.LayerNorm(self.semantic_dim)
        self.norm_prot_sem = nn.LayerNorm(self.semantic_dim)
        self.norm_drug_str = nn.LayerNorm(self.structural_dim)
        self.norm_prot_str = nn.LayerNorm(self.structural_dim)

        # Cross Attention Layers
        self.semantic_drug_cross_attn = nn.MultiheadAttention(
            self.semantic_dim, nhead, dropout=dropout, batch_first=True
        )
        self.semantic_protein_cross_attn = nn.MultiheadAttention(
            self.semantic_dim, nhead, dropout=dropout, batch_first=True
        )
        self.structural_drug_cross_attn = nn.MultiheadAttention(
            self.structural_dim, nhead, dropout=dropout, batch_first=True
        )
        self.structural_protein_cross_attn = nn.MultiheadAttention(
            self.structural_dim, nhead, dropout=dropout, batch_first=True
        )

        # Transformer Encoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=1)

        # Aggregators & Predictor
        self.drug_aggregator = LightweightAttention(embed_dim)
        self.protein_aggregator = LightweightAttention(embed_dim)

        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )

    def forward(self, drug_features, protein_features, drug_mask, protein_mask):
        # drug_features: (Batch, M, 1024)
        # protein_features: (Batch, N, 1024)
        
        # [修改点 2] 在分割特征之前，先加上位置编码
        # 这样位置信息会同时融入到语义部分(前512)和结构部分(后512)
        drug_features = self.pos_encoder(drug_features)
        protein_features = self.pos_encoder(protein_features)

        # 1. 分割为语义和结构部分
        drug_semantic = drug_features[:, :, :self.semantic_dim]
        drug_structural = drug_features[:, :, self.semantic_dim:]
        protein_semantic = protein_features[:, :, :self.semantic_dim]
        protein_structural = protein_features[:, :, self.semantic_dim:]

        # 2. 语义信息处理 (Cross Attn + Residual + Norm)
        drug_sem_attn_out, _ = self.semantic_drug_cross_attn(
            drug_semantic, protein_semantic, protein_semantic,
            key_padding_mask=protein_mask
        )
        drug_semantic_context = self.norm_drug_sem(
            drug_semantic + self.dropout_layer(drug_sem_attn_out)
        )

        prot_sem_attn_out, _ = self.semantic_protein_cross_attn(
            protein_semantic, drug_semantic, drug_semantic,
            key_padding_mask=drug_mask
        )
        protein_semantic_context = self.norm_prot_sem(
            protein_semantic + self.dropout_layer(prot_sem_attn_out)
        )

        # 3. 结构信息处理 (Cross Attn + Residual + Norm)
        drug_str_attn_out, _ = self.structural_drug_cross_attn(
            drug_structural, protein_structural, protein_structural,
            key_padding_mask=protein_mask
        )
        drug_structural_context = self.norm_drug_str(
            drug_structural + self.dropout_layer(drug_str_attn_out)
        )

        prot_str_attn_out, _ = self.structural_protein_cross_attn(
            protein_structural, drug_structural, drug_structural,
            key_padding_mask=drug_mask
        )
        protein_structural_context = self.norm_prot_str(
            protein_structural + self.dropout_layer(prot_str_attn_out)
        )

        # 4. 拼接
        drug_context = torch.cat([drug_semantic_context, drug_structural_context], dim=-1)
        protein_context = torch.cat([protein_semantic_context, protein_structural_context], dim=-1)

        # 5. Transformer Encoder
        drug_context_encoded = self.transformer_encoder(drug_context, src_key_padding_mask=drug_mask)
        protein_context_encoded = self.transformer_encoder(protein_context, src_key_padding_mask=protein_mask)

        # 6. 聚合
        drug_agg = self.drug_aggregator(drug_context_encoded, drug_mask)
        protein_agg = self.protein_aggregator(protein_context_encoded, protein_mask)

        # 7. 预测
        combined_features = torch.cat([drug_agg, protein_agg], dim=1)
        output = self.predictor(combined_features)

        return output.squeeze(-1)


# --- 3. 数据集与数据加载器 (Dataset and DataLoader) ---

class DPIDataset(Dataset):
    def __init__(self, pairs, drug_map, protein_map, drug_features, protein_features):
        self.pairs = pairs
        self.drug_map = drug_map
        self.protein_map = protein_map
        self.drug_features = torch.from_numpy(drug_features).float()
        self.protein_features = torch.from_numpy(protein_features).float()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        drug_id, protein_id, label = self.pairs[idx]
        sub_indices = self.drug_map[drug_id]
        dom_indices = self.protein_map[protein_id]
        drug_feat = self.drug_features[sub_indices]
        prot_feat = self.protein_features[dom_indices]
        return drug_feat, prot_feat, torch.tensor(label, dtype=torch.float)


def collate_fn(batch):
    drug_feats, prot_feats, labels = zip(*batch)
    drug_lens = torch.tensor([len(d) for d in drug_feats])
    prot_lens = torch.tensor([len(p) for p in prot_feats])
    drug_padded = pad_sequence(drug_feats, batch_first=True, padding_value=0)
    prot_padded = pad_sequence(prot_feats, batch_first=True, padding_value=0)
    drug_mask = torch.arange(drug_padded.size(1))[None, :] >= drug_lens[:, None]
    prot_mask = torch.arange(prot_padded.size(1))[None, :] >= prot_lens[:, None]
    labels = torch.stack(labels)
    return drug_padded, prot_padded, drug_mask, prot_mask, labels


# --- 4. 数据加载与预处理函数 ---

def load_data(config, fold):
    print(f"--- Loading data for fold {fold} ---")
    drug_features_all = np.load(config.DRUG_FEATURE_FILE)
    protein_features_all = np.load(config.PROTEIN_FEATURE_FILE)
    print(f"Loaded drug substructure features: {drug_features_all.shape}")
    print(f"Loaded protein domain features: {protein_features_all.shape}")

    drug_map_df = pd.read_csv(config.DRUG_MAP_FILE)
    protein_map_df = pd.read_csv(config.PROTEIN_MAP_FILE)

    drug_map_dict = {}
    for _, row in drug_map_df.dropna(subset=['drug_substructure']).iterrows():
        indices = [int(s.replace('sub_', '')) - 1 for s in row['drug_substructure'].split(',')]
        drug_map_dict[row['drug_id']] = indices

    protein_map_dict = {}
    for _, row in protein_map_df.dropna(subset=['protein_domain']).iterrows():
        indices = [int(d.replace('dom_', '')) - 1 for d in row['protein_domain'].split(',')]
        protein_map_dict[row['protein_id']] = indices

    drugbank_df = pd.read_csv(config.DRUGBANK_FILE, sep=' ', header=None,
                              names=['drug_id', 'protein_id', 'drug_smile', 'protein_sequence', 'label'])

    valid_pairs = []
    print("Filtering DrugBank pairs...")
    for _, row in drugbank_df.iterrows():
        d_id, p_id = row['drug_id'], row['protein_id']
        if (d_id in drug_map_dict) and (p_id in protein_map_dict):
            valid_pairs.append((d_id, p_id, row['label']))

    random.Random(42).shuffle(valid_pairs)
    fold_size = len(valid_pairs) // 5 
    folds = [valid_pairs[i * fold_size: (i + 1) * fold_size] for i in range(5)]
    validation_set = folds[fold]
    training_set = [pair for j, f in enumerate(folds) if j != fold for pair in f]
    print(f"Fold {fold + 1} - Training set size: {len(training_set)}, Validation set size: {len(validation_set)}")

    train_dataset = DPIDataset(training_set, drug_map_dict, protein_map_dict, drug_features_all, protein_features_all)
    val_dataset = DPIDataset(validation_set, drug_map_dict, protein_map_dict, drug_features_all, protein_features_all)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=config.NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader


# --- 5. 训练与评估函数 ---

def calculate_metrics(all_labels, all_preds_prob, all_preds_class):
    acc = accuracy_score(all_labels, all_preds_class)
    prec = precision_score(all_labels, all_preds_class)
    rec = recall_score(all_labels, all_preds_class)
    try:
        auc = roc_auc_score(all_labels, all_preds_prob)
    except ValueError:
        auc = 0.5 
    try:
        aupr = average_precision_score(all_labels, all_preds_prob)
    except ValueError:
        aupr = 0
    f1 = f1_score(all_labels, all_preds_class)
    return acc, prec, rec, auc, aupr, f1


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for drug_padded, prot_padded, drug_mask, prot_mask, labels in tqdm(dataloader, desc="Training"):
        drug_padded, prot_padded = drug_padded.to(device), prot_padded.to(device)
        drug_mask, prot_mask = drug_mask.to(device), prot_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(drug_padded, prot_padded, drug_mask, prot_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds_prob = []

    with torch.no_grad():
        for drug_padded, prot_padded, drug_mask, prot_mask, labels in tqdm(dataloader, desc="Evaluating"):
            drug_padded, prot_padded = drug_padded.to(device), prot_padded.to(device)
            drug_mask, prot_mask = drug_mask.to(device), prot_mask.to(device)
            labels = labels.to(device)

            outputs = model(drug_padded, prot_padded, drug_mask, prot_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds_prob = torch.sigmoid(outputs).cpu().numpy()
            all_preds_prob.extend(preds_prob)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_labels = np.array(all_labels)
    all_preds_prob = np.array(all_preds_prob)
    all_preds_class = (all_preds_prob > 0.5).astype(int)
    acc, prec, rec, auc, aupr, f1 = calculate_metrics(all_labels, all_preds_prob, all_preds_class)
    return avg_loss, (acc, prec, rec, auc, aupr, f1)


# --- 6. 主执行函数 ---

def main():
    config = Config()
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    best_fold_metrics = []

    for fold in range(5):
        print(f"\n--- Starting fold {fold + 1} ---")
        train_loader, val_loader = load_data(config, fold)

        model = DPIPredictor(
            embed_dim=config.EMBED_DIM,
            nhead=config.ATTN_HEADS,
            dropout=config.DROPOUT
        ).to(config.DEVICE)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        print(f"\nModel initialized. Starting training...")

        best_val_aupr = 0
        best_val_metrics = None
        patience_counter = 0
        
        os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)

        with open(config.LOG_FILE, 'a') as log_f:
            if fold == 0:
                log_f.write("Epoch,Train_Loss,Val_Loss,Val_Acc,Val_Prec,Val_Rec,Val_AUC,Val_AUPR,Val_F1\n")

            for epoch in range(config.MAX_EPOCHS):
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
                val_loss, (val_acc, val_prec, val_rec, val_auc, val_aupr, val_f1) = evaluate(model, val_loader, criterion, config.DEVICE)

                print(f"Epoch {epoch + 1}/{config.MAX_EPOCHS} | Val Loss: {val_loss:.4f} | "
                      f"Val AUC: {val_auc:.4f} | Val AUPR: {val_aupr:.4f}") 

                log_line = f"{fold}_{epoch + 1},{train_loss:.6f},{val_loss:.6f},{val_acc:.4f},{val_prec:.4f},{val_rec:.4f},{val_auc:.4f},{val_aupr:.4f},{val_f1:.4f}\n"
                log_f.write(log_line)
                log_f.flush()

                if val_aupr > best_val_aupr:
                    best_val_aupr = val_aupr
                    best_val_metrics = (val_acc, val_prec, val_rec, val_auc, val_aupr, val_f1)
                    patience_counter = 0
                    torch.save(model.state_dict(), config.MODEL_SAVE_PATH.replace('.pth', f'_fold{fold+1}.pth'))
                    print(f"Validation AUPR improved to {best_val_aupr:.4f}. Model saved.")
                else:
                    patience_counter += 1

                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1}.")
                    break

        if best_val_metrics is not None:
            best_fold_metrics.append(best_val_metrics)
        else:
             best_fold_metrics.append((val_acc, val_prec, val_rec, val_auc, val_aupr, val_f1))

    avg_metrics = {
        "acc": np.mean([metrics[0] for metrics in best_fold_metrics]),
        "prec": np.mean([metrics[1] for metrics in best_fold_metrics]),
        "rec": np.mean([metrics[2] for metrics in best_fold_metrics]),
        "auc": np.mean([metrics[3] for metrics in best_fold_metrics]),
        "aupr": np.mean([metrics[4] for metrics in best_fold_metrics]),
        "f1": np.mean([metrics[5] for metrics in best_fold_metrics])
    }
    std_metrics = {
        "acc": np.std([metrics[0] for metrics in best_fold_metrics]),
        "prec": np.std([metrics[1] for metrics in best_fold_metrics]),
        "rec": np.std([metrics[2] for metrics in best_fold_metrics]),
        "auc": np.std([metrics[3] for metrics in best_fold_metrics]),
        "aupr": np.std([metrics[4] for metrics in best_fold_metrics]),
        "f1": np.std([metrics[5] for metrics in best_fold_metrics])
    }

    print("\n--- Final Results (average of all folds) ---")
    print(f"Avg AUC      : {avg_metrics['auc']:.4f}")
    print(f"Avg AUPR     : {avg_metrics['aupr']:.4f}")
    
    with open(config.LOG_FILE, 'a') as log_f:
        log_f.write("\n--- Final Results (average of all folds) ---\n")
        log_f.write(f"Avg_AUC,{avg_metrics['auc']:.4f}\n")
        log_f.write(f"Avg_AUPR,{avg_metrics['aupr']:.4f}\n")

if __name__ == '__main__':
    main()
