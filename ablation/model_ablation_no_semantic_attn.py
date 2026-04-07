import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --- 1. 超参数与配置 (Hyperparameters and Configuration) ---
class Config:
    # 文件路径
    BASE_PATH = '/media/4T2/zc/Drug-Protein/datasets/KIBA/final'
    DRUGBANK_FILE = os.path.join(BASE_PATH, 'KIBA.txt')
    DRUG_MAP_FILE = os.path.join(BASE_PATH, 'drug_mapping.csv')
    PROTEIN_MAP_FILE = os.path.join(BASE_PATH, 'protein_mapping.csv')
    DRUG_FEATURE_FILE = os.path.join(BASE_PATH, 'drug_merge_1024.npy')
    PROTEIN_FEATURE_FILE = os.path.join(BASE_PATH, 'protein_merge_1024.npy')

    # 模型保存与日志
    MODEL_SAVE_PATH = '/media/4T2/lmd/Qwen2.5/drug/ablation/no_semantic_attn/KIBA_no_semantic_attn.pth'
    LOG_FILE = '/media/4T2/lmd/Qwen2.5/drug/ablation/no_semantic_attn/KIBA_no_semantic_attn.txt'

    # 模型参数
    EMBED_DIM = 1024
    ATTN_HEADS = 8  # 交叉注意力和Transformer的头数
    DROPOUT = 0.1

    # 训练参数
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 200
    EARLY_STOPPING_PATIENCE = 30
    NUM_WORKERS = 16  # 用于DataLoader的多进程数据加载

# --- 2. 模型架构定义 (Model Architecture) ---
class LightweightAttention(nn.Module):
    """
    轻量级自注意力机制，用于聚合序列信息
    输入: (Batch, Seq_len, Dim)
    输出: (Batch, Dim)
    """
    def __init__(self, input_dim):
        super(LightweightAttention, self).__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x, mask):
        attention_scores = self.attention_layer(x).squeeze(-1)  # (Batch, Seq_len)
        attention_scores.masked_fill_(mask == 0, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)  # (Batch, Seq_len, 1)
        context_vector = torch.sum(x * attention_weights, dim=1)  # (Batch, Dim)
        return context_vector

class DPIPredictor(nn.Module):
    def __init__(self, embed_dim, nhead, dropout=0.1):
        super(DPIPredictor, self).__init__()
        self.embed_dim = embed_dim  # 1024
        self.structural_dim = embed_dim // 2  # 512

        # 结构信息的双向交叉注意力
        self.structural_drug_cross_attn = nn.MultiheadAttention(
            self.structural_dim, nhead, dropout=dropout, batch_first=True
        )
        self.structural_protein_cross_attn = nn.MultiheadAttention(
            self.structural_dim, nhead, dropout=dropout, batch_first=True
        )

        # Transformer编码器层
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=1)

        # 轻量级注意力聚合模块
        self.drug_aggregator = LightweightAttention(embed_dim)
        self.protein_aggregator = LightweightAttention(embed_dim)

        # 最终预测的MLP
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
        # 输入：
        # drug_features: (Batch, M, 1024)
        # protein_features: (Batch, N, 1024)
        # drug_mask: (Batch, M), protein_mask: (Batch, N), True表示padding

        # 1. 分割为语义和结构部分
        drug_semantic = drug_features[:, :, :self.structural_dim]  # (Batch, M, 512)
        drug_structural = drug_features[:, :, self.structural_dim:]  # (Batch, M, 512)
        protein_semantic = protein_features[:, :, :self.structural_dim]  # (Batch, N, 512)
        protein_structural = protein_features[:, :, self.structural_dim:]  # (Batch, N, 512)

        # 2. 结构信息的双向交叉注意力
        drug_structural_context, _ = self.structural_drug_cross_attn(
            drug_structural, protein_structural, protein_structural,
            key_padding_mask=protein_mask
        )
        protein_structural_context, _ = self.structural_protein_cross_attn(
            protein_structural, drug_structural, drug_structural,
            key_padding_mask=drug_mask
        )

        # 3. 拼接语义（未处理）和结构上下文
        drug_context = torch.cat([drug_semantic, drug_structural_context], dim=-1)  # (Batch, M, 1024)
        protein_context = torch.cat([protein_semantic, protein_structural_context], dim=-1)  # (Batch, N, 1024)

        # 4. Transformer整合全局信息
        drug_context_encoded = self.transformer_encoder(drug_context, src_key_padding_mask=drug_mask)
        protein_context_encoded = self.transformer_encoder(protein_context, src_key_padding_mask=protein_mask)

        # 5. 轻量级注意力聚合
        drug_agg = self.drug_aggregator(drug_context_encoded, ~drug_mask)
        protein_agg = self.protein_aggregator(protein_context_encoded, ~protein_mask)

        # 6. 拼接与预测
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
def load_data(config):
    print("--- Loading and Preprocessing Data ---")
    drug_features_all = np.load(config.DRUG_FEATURE_FILE)
    protein_features_all = np.load(config.PROTEIN_FEATURE_FILE)
    print(f"Loaded drug substructure features: {drug_features_all.shape}")
    print(f"Loaded protein domain features: {protein_features_all.shape}")

    drug_map_df = pd.read_csv(config.DRUG_MAP_FILE)
    protein_map_df = pd.read_csv(config.PROTEIN_MAP_FILE)
    bad_drug_ids = set(drug_map_df[drug_map_df['drug_substructure'].isna()]['drug_id'])
    print(f"Drugs with NaN substructures (to be excluded): {bad_drug_ids}")
    bad_protein_ids = set(protein_map_df[protein_map_df['protein_domain'].isna()]['protein_id'])
    print(f"Proteins with NaN domains (to be excluded): {bad_protein_ids}")

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
    for _, row in tqdm(drugbank_df.iterrows(), total=len(drugbank_df)):
        d_id, p_id = row['drug_id'], row['protein_id']
        if (d_id in drug_map_dict) and (p_id in protein_map_dict) and \
                (d_id not in bad_drug_ids) and (p_id not in bad_protein_ids):
            valid_pairs.append((d_id, p_id, row['label']))

    print(f"Original pairs: {len(drugbank_df)}, Valid pairs after filtering: {len(valid_pairs)}")

    train_val_pairs, test_pairs = train_test_split(valid_pairs, test_size=0.1, random_state=42,
                                                   stratify=[p[2] for p in valid_pairs])
    train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=1/9, random_state=42,
                                              stratify=[p[2] for p in train_val_pairs])

    print(f"Train samples: {len(train_pairs)}, Validation samples: {len(val_pairs)}, Test samples: {len(test_pairs)}")

    train_dataset = DPIDataset(train_pairs, drug_map_dict, protein_map_dict, drug_features_all, protein_features_all)
    val_dataset = DPIDataset(val_pairs, drug_map_dict, protein_map_dict, drug_features_all, protein_features_all)
    test_dataset = DPIDataset(test_pairs, drug_map_dict, protein_map_dict, drug_features_all, protein_features_all)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                             collate_fn=collate_fn, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                           collate_fn=collate_fn, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=config.NUM_WORKERS)

    return train_loader, val_loader, test_loader

# --- 5. 训练与评估函数 ---
def calculate_metrics(all_labels, all_preds_prob, all_preds_class):
    acc = accuracy_score(all_labels, all_preds_class)
    prec = precision_score(all_labels, all_preds_class)
    rec = recall_score(all_labels, all_preds_class)
    auc = roc_auc_score(all_labels, all_preds_prob)
    aupr = average_precision_score(all_labels, all_preds_prob)
    return acc, prec, rec, auc, aupr

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

    metrics = calculate_metrics(all_labels, all_preds_prob, all_preds_class)
    return avg_loss, metrics

# --- 6. 主执行函数 ---
def main():
    config = Config()
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    train_loader, val_loader, test_loader = load_data(config)
    model = DPIPredictor(
        embed_dim=config.EMBED_DIM,
        nhead=config.ATTN_HEADS,
        dropout=config.DROPOUT
    ).to(config.DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"\nModel initialized on {config.DEVICE}. Starting training...")

    best_val_aupr = 0
    patience_counter = 0

    with open(config.LOG_FILE, 'w') as log_f:
        log_f.write("Epoch,Train_Loss,Val_Loss,Val_Acc,Val_Prec,Val_Rec,Val_AUC,Val_AUPR\n")

        for epoch in range(config.MAX_EPOCHS):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
            val_loss, (val_acc, val_prec, val_rec, val_auc, val_aupr) = evaluate(model, val_loader, criterion,
                                                                                 config.DEVICE)

            print(f"Epoch {epoch + 1}/{config.MAX_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} | Val AUPR: {val_aupr:.4f}")

            log_line = f"{epoch + 1},{train_loss:.6f},{val_loss:.6f},{val_acc:.4f},{val_prec:.4f},{val_rec:.4f},{val_auc:.4f},{val_aupr:.4f}\n"
            log_f.write(log_line)
            log_f.flush()

            if val_aupr > best_val_aupr:
                best_val_aupr = val_aupr
                patience_counter = 0
                torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
                print(f"Validation AUPR improved to {best_val_aupr:.4f}. Model saved to {config.MODEL_SAVE_PATH}")
            else:
                patience_counter += 1
                print(f"No improvement in validation AUPR for {patience_counter} epochs.")

            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch + 1} as validation AUPR did not improve for {config.EARLY_STOPPING_PATIENCE} epochs.")
                break

    print("\n--- Training Finished ---")
    print("\n--- Evaluating on Test Set with Best Model ---")
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    test_loss, (test_acc, test_prec, test_rec, test_auc, test_aupr) = evaluate(model, test_loader, criterion,
                                                                               config.DEVICE)

    print("\n--- Test Set Performance ---")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall   : {test_rec:.4f}")
    print(f"AUC      : {test_auc:.4f}")
    print(f"AUPR     : {test_aupr:.4f}")

    with open(config.LOG_FILE, 'a') as log_f:
        log_f.write("\n--- Test Set Performance ---\n")
        log_f.write(f"Test_Loss,{test_loss:.6f}\n")
        log_f.write(f"Test_Accuracy,{test_acc:.4f}\n")
        log_f.write(f"Test_Precision,{test_prec:.4f}\n")
        log_f.write(f"Test_Recall,{test_rec:.4f}\n")
        log_f.write(f"Test_AUC,{test_auc:.4f}\n")
        log_f.write(f"Test_AUPR,{test_aupr:.4f}\n")

if __name__ == '__main__':
    main()