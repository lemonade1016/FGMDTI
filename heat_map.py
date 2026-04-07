import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# --- 1. 超参数与配置 ---
class Config:
    BASE_PATH = '/media/4T2/zc/Drug-Protein/datasets/drugbank/final'
    DRUGBANK_FILE = os.path.join(BASE_PATH, 'DrugBank.txt')
    DRUG_MAP_FILE = os.path.join(BASE_PATH, 'drug_mapping.csv')
    PROTEIN_MAP_FILE = os.path.join(BASE_PATH, 'protein_mapping.csv')
    DRUG_FEATURE_FILE = os.path.join(BASE_PATH, 'drug_merge_1024.npy')
    PROTEIN_FEATURE_FILE = os.path.join(BASE_PATH, 'protein_merge_1024.npy')
    MODEL_SAVE_PATH = os.path.join(BASE_PATH, 'lmd_model/best_model.pth')
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EMBED_DIM = 1024
    ATTN_HEADS = 8
    DROPOUT = 0.1

# --- 2. 模型架构定义 ---
class LightweightAttention(nn.Module):
    def __init__(self, input_dim):
        super(LightweightAttention, self).__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x, mask):
        attention_scores = self.attention_layer(x).squeeze(-1)
        attention_scores.masked_fill_(mask == 0, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
        context_vector = torch.sum(x * attention_weights, dim=1)
        return context_vector

class DPIPredictor(nn.Module):
    def __init__(self, embed_dim, nhead, dropout=0.1):
        super(DPIPredictor, self).__init__()
        self.embed_dim = embed_dim
        self.drug_cross_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.protein_cross_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=1)
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
        drug_context, drug_attn_weights = self.drug_cross_attn(
            drug_features, protein_features, protein_features,
            key_padding_mask=protein_mask, need_weights=True
        )
        protein_context, protein_attn_weights = self.protein_cross_attn(
            protein_features, drug_features, drug_features,
            key_padding_mask=drug_mask, need_weights=True
        )
        drug_context_encoded = self.transformer_encoder(drug_context, src_key_padding_mask=drug_mask)
        protein_context_encoded = self.transformer_encoder(protein_context, src_key_padding_mask=protein_mask)
        drug_agg = self.drug_aggregator(drug_context_encoded, ~drug_mask)
        protein_agg = self.protein_aggregator(protein_context_encoded, ~protein_mask)
        combined_features = torch.cat([drug_agg, protein_agg], dim=1)
        output = self.predictor(combined_features)
        return output.squeeze(-1), drug_attn_weights, protein_attn_weights

# --- 3. 生成均值注意力图 ---
def generate_mean_attention_maps(config, model, drug_features_all, protein_features_all):
    drugbank_df = pd.read_csv(config.DRUGBANK_FILE, sep=' ', header=None,
                              names=['drug_id', 'protein_id', 'drug_smile', 'protein_sequence', 'label'])
    drug_map_df = pd.read_csv(config.DRUG_MAP_FILE)
    protein_map_df = pd.read_csv(config.PROTEIN_MAP_FILE)

    # 创建子结构和结构域的映射
    drug_sub_smiles_dict = {}
    for _, row in drug_map_df.iterrows():
        if pd.notna(row['drug_substructure']) and pd.notna(row['substructure_seq']):
            drug_sub_smiles_dict[row['drug_id']] = dict(zip(
                row['drug_substructure'].split(','),
                row['substructure_seq'].split(',')
            ))

    protein_dom_seq_dict = {}
    for _, row in protein_map_df.iterrows():
        if pd.notna(row['protein_domain']) and pd.notna(row['domain_seq']):
            protein_dom_seq_dict[row['protein_id']] = dict(zip(
                row['protein_domain'].split(','),
                row['domain_seq'].split(',')
            ))

    # 打开输出文件
    with open(os.path.join(config.BASE_PATH, 'avg_attention_maps.txt'), 'w') as txt_file, \
            open(os.path.join(config.BASE_PATH, 'avg_attention_maps.csv'), 'w') as csv_file:
        for _, row in tqdm(drugbank_df.iterrows(), total=len(drugbank_df), desc="Generating Mean Attention Maps"):
            drug_id, protein_id = row['drug_id'], row['protein_id']
            # 跳过无效的drug_id或protein_id
            if drug_id not in drug_sub_smiles_dict or protein_id not in protein_dom_seq_dict:
                continue
            # 获取子结构和结构域
            substructures = list(drug_sub_smiles_dict[drug_id].keys())
            domains = list(protein_dom_seq_dict[protein_id].keys())
            sub_indices = [int(s.replace('sub_', '')) - 1 for s in substructures]
            dom_indices = [int(d.replace('dom_', '')) - 1 for d in domains]
            # 获取特征
            drug_feat = torch.from_numpy(drug_features_all[sub_indices]).float().unsqueeze(0).to(config.DEVICE)
            prot_feat = torch.from_numpy(protein_features_all[dom_indices]).float().unsqueeze(0).to(config.DEVICE)
            drug_mask = torch.zeros(1, len(substructures), dtype=torch.bool).to(config.DEVICE)
            prot_mask = torch.zeros(1, len(domains), dtype=torch.bool).to(config.DEVICE)
            # 获取注意力权重
            with torch.no_grad():
                _, drug_attn_weights, protein_attn_weights = model(drug_feat, prot_feat, drug_mask, prot_mask)
            # 转换为numpy数组
            drug_attn_matrix = drug_attn_weights.squeeze(0).cpu().numpy()  # Shape: (M, N)
            protein_attn_matrix = protein_attn_weights.squeeze(0).cpu().numpy()  # Shape: (N, M)
            # 计算均值注意力图：将protein_attn_matrix转置为(M, N)后取平均
            mean_attn_matrix = (drug_attn_matrix + protein_attn_matrix.T) / 2  # Shape: (M, N)
            sub_smiles = [drug_sub_smiles_dict[drug_id][sub] for sub in substructures]
            dom_seqs = [protein_dom_seq_dict[protein_id][dom] for dom in domains]
            # 写入TXT文件
            txt_file.write(f"Fragment-DomainInteraction:{drug_id}x{protein_id}\n")
            txt_file.write("Substructure_SMILES\t" + "\t".join(dom_seqs) + "\n")
            for i, smile in enumerate(sub_smiles):
                txt_file.write(smile + "\t" + "\t".join(map(str, mean_attn_matrix[i])) + "\n")
            txt_file.write("\n")
            # 写入CSV文件
            csv_file.write(f"Fragment-DomainInteraction:{drug_id}x{protein_id}\n")
            csv_file.write("Substructure_SMILES," + ",".join(dom_seqs) + "\n")
            for i, smile in enumerate(sub_smiles):
                csv_file.write(smile + "," + ",".join(map(str, mean_attn_matrix[i])) + "\n")
            csv_file.write("\n")

# --- 4. 主执行函数 ---
def main():
    config = Config()
    # 加载特征
    drug_features_all = np.load(config.DRUG_FEATURE_FILE)
    protein_features_all = np.load(config.PROTEIN_FEATURE_FILE)
    print(f"Loaded drug substructure features: {drug_features_all.shape}")
    print(f"Loaded protein domain features: {protein_features_all.shape}")
    # 加载模型
    model = DPIPredictor(embed_dim=config.EMBED_DIM, nhead=config.ATTN_HEADS, dropout=config.DROPOUT).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    model.eval()
    print(f"Model loaded from {config.MODEL_SAVE_PATH}")
    # 生成均值注意力图
    generate_mean_attention_maps(config, model, drug_features_all, protein_features_all)
    print("Mean attention maps saved to 'attention_maps_mean.txt' and 'attention_maps_mean.csv'")

if __name__ == '__main__':
    main()
