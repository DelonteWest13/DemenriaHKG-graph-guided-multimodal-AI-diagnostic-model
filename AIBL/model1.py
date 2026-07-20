import torch
import torch.nn as nn
import os
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import math
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def center_crop_3d(tensor, size):
    depth, height, width = tensor.shape
    target_depth, target_height, target_width = size

    start_depth = (depth - target_depth) // 2
    start_height = (height - target_height) // 2
    start_width = (width - target_width) // 2

    end_depth = start_depth + target_depth
    end_height = start_height + target_height
    end_width = start_width + target_width

    return tensor[start_depth:end_depth, start_height:end_height, start_width:end_width]


class NiiDataset(Dataset):
    def __init__(self, nii_file_list):
        """
        nii_file_list: 完整的 NII 路径列表，顺序与你的EHR严格一致。
        """
        self.file_list = nii_file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        img = nib.load(file_path)
        img_data = img.get_fdata()
        img_tensor = torch.from_numpy(img_data).float()
        cropped_tensor = center_crop_3d(img_tensor, (64, 64, 64)) # 按你需求可改
        input_tensor = cropped_tensor.unsqueeze(0)
        return input_tensor



class CNN_3D(nn.Module):
    def __init__(self, num_class=1):  # num_class
        super().__init__()
        self.features = nn.Sequential(
            # 1 128 128 128 3 1 2
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), #1 64 64 64
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            # 1 32 32 32
            nn.Linear(32 * 16 * 16 * 16, 64),  #32 24
            nn.ReLU(),
            nn.Linear(64, num_class)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 定义神经网络模型
class NeuralNet(nn.Module):
    def __init__(self,embedding):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding, 32),
            nn.ReLU(),    # 添加非线性激活函数
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # 1wei
        )
    
    def forward(self, x):
        return self.layers(x)

# 4.11双模态 模型定义
class DualTransformer(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, num_layers=2):
        super().__init__()
        # 模态编码器 (输出形状: [batch, embed_dim])
        self.ehr_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )
        self.img_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )
        
        # 模态 A (ehr) 的 Q、K、V 线性层
        self.ehr_q = nn.Linear(embed_dim, embed_dim)
        self.ehr_k = nn.Linear(embed_dim, embed_dim)
        self.ehr_v = nn.Linear(embed_dim, embed_dim)
        
        # 模态 B (img) 的 Q、K、V 线性层
        self.img_q = nn.Linear(embed_dim, embed_dim)
        self.img_k = nn.Linear(embed_dim, embed_dim)
        self.img_v = nn.Linear(embed_dim, embed_dim)
        
        # 交叉注意力机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 将 ab_combined 的维度从 2 * embed_dim 调整为 embed_dim
        self.ab_proj = nn.Linear(2 * embed_dim, embed_dim)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),  # 输入维度改为 2 * embed_dim
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )
        
    def forward(self, x):
        # 输入分解
        ehr = x[:, 0].unsqueeze(1)  # [batch, 1]
        img = x[:, 1].unsqueeze(1)
        
        # 模态编码
        ehr_feat = self.ehr_encoder(ehr).unsqueeze(1)  # [batch, 1, embed_dim]
        img_feat = self.img_encoder(img).unsqueeze(1)
        
        # 模态 A (ehr) 的 Q、K、V
        Qa = self.ehr_q(ehr_feat)  # [batch, 1, embed_dim]
        Ka = self.ehr_k(ehr_feat)
        Va = self.ehr_v(ehr_feat)
        
        # 模态 B (img) 的 Q、K、V
        Qb = self.img_q(img_feat)  # [batch, 1, embed_dim]
        Kb = self.img_k(img_feat)
        Vb = self.img_v(img_feat)
        
        # 交叉注意力机制
        # Qa 与 Kb、Vb 进行注意力计算
        ehr_img_attn, _ = self.cross_attention(
            query=Qa,  # [batch, 1, embed_dim]
            key=Kb,    # [batch, 1, embed_dim]
            value=Vb   # [batch, 1, embed_dim]
        )
        
        # Qb 与 Ka、Va 进行注意力计算
        img_ehr_attn, _ = self.cross_attention(
            query=Qb,  # [batch, 1, embed_dim]
            key=Ka,    # [batch, 1, embed_dim]
            value=Va   # [batch, 1, embed_dim]
        )
        
        # 拼接交叉注意力结果
        ab_combined = torch.cat([ehr_img_attn.squeeze(1), img_ehr_attn.squeeze(1)], dim=1)  # [batch, 2 * embed_dim]
        
        # 输入到融合层
        return self.fusion(ab_combined)




#4.11 三模态
class MultiModalTransformer(nn.Module):
    def __init__(self, embed_dim=16, num_heads=2, num_layers=2):
        super().__init__()
        # 模态编码器 (输出形状: [batch, embed_dim])
        self.ehr_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )
        self.img_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )
        self.bio_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )
        

        self.bio_q = nn.Linear(embed_dim, embed_dim)
        self.bio_k = nn.Linear(embed_dim, embed_dim)
        self.bio_v = nn.Linear(embed_dim, embed_dim)

        self.ehr_q = nn.Linear(embed_dim, embed_dim)
        self.ehr_k = nn.Linear(embed_dim, embed_dim)
        self.ehr_v = nn.Linear(embed_dim, embed_dim)

        self.img_q = nn.Linear(embed_dim, embed_dim)
        self.img_k = nn.Linear(embed_dim, embed_dim)
        self.img_v = nn.Linear(embed_dim, embed_dim)
        
        # 交叉注意力机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 将 ab_combined 的维度从 2 * embed_dim 调整为 embed_dim
        self.ab_proj = nn.Linear(2 * embed_dim, embed_dim)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),  # 输入维度改为 2 * embed_dim
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
        
    def forward(self, x):
        # 输入分解
        ehr = x[:, 0].unsqueeze(1)  # [batch, 1]
        img = x[:, 1].unsqueeze(1)
        bio = x[:, 2].unsqueeze(1)
        
        # 模态编码
        ehr_feat = self.ehr_encoder(ehr).unsqueeze(1)  # [batch, 1, embed_dim]
        img_feat = self.img_encoder(img).unsqueeze(1)
        bio_feat = self.bio_encoder(bio).unsqueeze(1)
 
        Qa = self.bio_q(bio_feat)  # [batch, 1, embed_dim]
        Ka = self.bio_k(bio_feat)
        Va = self.bio_v(bio_feat)

        Qb = self.ehr_q(ehr_feat)  # [batch, 1, embed_dim]
        Kb = self.ehr_k(ehr_feat)
        Vb = self.ehr_v(ehr_feat)

        Qc = self.img_q(img_feat)  # [batch, 1, embed_dim]
        Kc = self.img_k(img_feat)
        Vc = self.img_v(img_feat)
        
        # 交叉注意力机制
        # Qa 与 Kb、Vb 进行注意力计算
        bio_ehr_attn, _ = self.cross_attention(
            query=Qa,  # [batch, 1, embed_dim]
            key=Kb,    # [batch, 1, embed_dim]
            value=Vb   # [batch, 1, embed_dim]
        )
        
        # Qb 与 Ka、Va 进行注意力计算
        ehr_bio_attn, _ = self.cross_attention(
            query=Qb,  # [batch, 1, embed_dim]
            key=Ka,    # [batch, 1, embed_dim]
            value=Va   # [batch, 1, embed_dim]
        )

        ab_combined = torch.cat([bio_ehr_attn, ehr_bio_attn], dim=1)  # [batch, 2, embed_dim]
        ab_combined = ab_combined.view(ab_combined.size(0), -1)  # [batch, 2 * embed_dim]
        ab_combined = self.ab_proj(ab_combined)  # [batch, embed_dim]
        ab_combined = ab_combined.unsqueeze(1)  # [batch, 1, embed_dim]
        
        # 模态 C 进行交叉注意力
        img_ab_attn, _ = self.cross_attention(
            query=Qc,  # [batch, 1, embed_dim]
            key=ab_combined,  # [batch, 1, embed_dim]
            value=ab_combined  # [batch, 1, embed_dim]
        )

        final_combined = torch.cat([ab_combined.squeeze(1), img_ab_attn.squeeze(1)], dim=1)  # [batch, 2 * embed_dim]
        
        # 输入到融合层
        return self.fusion(final_combined)


class TransEModel(nn.Module):
    def __init__(self, num_entities, num_relations, embed_dim):
        super().__init__()
        self.ent_embeddings = nn.Embedding(num_entities, embed_dim)
        self.rel_embeddings = nn.Embedding(num_relations, embed_dim) 
        self.zero_const = nn.Parameter(torch.zeros(1)) 
        self.pi_const = nn.Parameter(torch.tensor(3.14159)) 

    def forward(self, h, r, t):
        h_embed = self.ent_embeddings(h)
        r_embed = self.rel_embeddings(r)
        t_embed = self.ent_embeddings(t)
        return h_embed, r_embed, t_embed


#KG嵌入MML的Transformer模型定义，适用于TransE/DistMult
class KGMultiModalTransformer_old(nn.Module):
    def __init__(self, embed_dim=16, num_heads=2, num_layers=2, transe_embed_dim=32):
        super().__init__()
        # 模态编码器
        self.ehr_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )
        self.img_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )
        self.bio_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )

        # TransE 嵌入投影层
        self.transe_proj = nn.Linear(transe_embed_dim, embed_dim)

        # 其他部分保持不变
        self.bio_q = nn.Linear(embed_dim, embed_dim)
        self.bio_k = nn.Linear(embed_dim, embed_dim)
        self.bio_v = nn.Linear(embed_dim, embed_dim)

        self.ehr_q = nn.Linear(embed_dim, embed_dim)
        self.ehr_k = nn.Linear(embed_dim, embed_dim)
        self.ehr_v = nn.Linear(embed_dim, embed_dim)

        self.img_q = nn.Linear(embed_dim, embed_dim)
        self.img_k = nn.Linear(embed_dim, embed_dim)
        self.img_v = nn.Linear(embed_dim, embed_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.ab_proj = nn.Linear(4 * embed_dim, embed_dim)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, 64),  # 输入维度改为 3 * embed_dim
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )

    def forward(self, x, transe_embed):
        # 输入分解
        ehr = x[:, 0].unsqueeze(1)  # [batch, 1]
        img = x[:, 1].unsqueeze(1)
        bio = x[:, 2].unsqueeze(1)

        # 模态编码
        ehr_feat = self.ehr_encoder(ehr).unsqueeze(1)  # [batch, 1, embed_dim]
        img_feat = self.img_encoder(img).unsqueeze(1)
        bio_feat = self.bio_encoder(bio).unsqueeze(1)

        # TransE 嵌入投影
        transe_feat = self.transe_proj(transe_embed).unsqueeze(1)  # [batch, 1, embed_dim]

        # 将 TransE 嵌入与其他模态特征拼接
        bio_feat = torch.cat([bio_feat, transe_feat], dim=1)  # [batch, 2, embed_dim]
        ehr_feat = torch.cat([ehr_feat, transe_feat], dim=1)
        img_feat = torch.cat([img_feat, transe_feat], dim=1)


        Qa = self.bio_q(bio_feat)  # [batch, 2, embed_dim]
        Ka = self.bio_k(bio_feat)
        Va = self.bio_v(bio_feat)

        Qb = self.ehr_q(ehr_feat)  # [batch, 2, embed_dim]
        Kb = self.ehr_k(ehr_feat)
        Vb = self.ehr_v(ehr_feat)

        Qc = self.img_q(img_feat)  # [batch, 2, embed_dim]
        Kc = self.img_k(img_feat)
        Vc = self.img_v(img_feat)

    # 交叉注意力机制
        bio_ehr_attn, _ = self.cross_attention(query=Qa, key=Kb, value=Vb)  # [batch, 2, embed_dim]
        ehr_bio_attn, _ = self.cross_attention(query=Qb, key=Ka, value=Va)  # [batch, 2, embed_dim]

    # 拼接交叉注意力结果
        ab_combined = torch.cat([bio_ehr_attn, ehr_bio_attn], dim=1)  # [batch, 4, embed_dim]
        ab_combined = ab_combined.view(ab_combined.size(0), -1)  # [batch, 4 * embed_dim]
        ab_combined = self.ab_proj(ab_combined).unsqueeze(1)  # [batch, 1, embed_dim]
    
    # 模态 C 进行交叉注意力
        img_ab_attn, _ = self.cross_attention(query=Qc, key=ab_combined, value=ab_combined)  # [batch, 2, embed_dim]

        final_combined = torch.cat(
            [
                ab_combined.squeeze(1),  
                img_ab_attn.mean(dim=1),  
                transe_feat.squeeze(1)   
            ],
            dim=1
        )  
        return self.fusion(final_combined)  # [batch, 3]



#4.3针对 AIBL 数据集定制的多模态协同交互网络
class KGMultiModalTransformer(nn.Module):
    """
    针对 AIBL 数据集定制的：多模态协同交互网络
    核心升级：引入 EHR 局部旁路机制，隔离非特异性图谱检索噪音；序列长度参数化。
    """
    def __init__(self, ehr_dim=2, img_dim=64, bio_dim=5, embed_dim=128, 
                 num_heads=2, num_layers=2, transe_embed_dim=128, max_seq_len=20):
        super().__init__()
        
        # AIBL 数据集特征维度设定
        self.ehr_dim = ehr_dim
        self.img_dim = img_dim
        self.bio_dim = bio_dim
        self.max_seq_len = max_seq_len  # 显式保存图谱序列长度
        
        # ==========================================
        # 1. 跨模态解耦编码器
        # ==========================================
        self.ehr_encoder = nn.Sequential(
            nn.Linear(ehr_dim, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
        
        self.img_encoder = nn.Sequential(
            nn.Linear(img_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
        
        self.bio_encoder = nn.Sequential(
            nn.Linear(bio_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )

        # ==========================================
        # 2. 知识流形投影层
        # ==========================================
        self.transe_proj = nn.Sequential(
            nn.Linear(transe_embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, embed_dim)
        )

        # ==========================================
        # 3. 模态感知知识检索引擎 (EHR 已移除)
        # ==========================================
        self.bio_kg_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.2)
        self.img_kg_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.2)

        self.norm_bio_kg = nn.LayerNorm(embed_dim)
        self.norm_img_kg = nn.LayerNorm(embed_dim)

        # ==========================================
        # 4. 线性投影层 (准备底层交互)
        # ==========================================
        self.bio_q = nn.Linear(embed_dim, embed_dim)
        self.bio_k = nn.Linear(embed_dim, embed_dim)
        self.bio_v = nn.Linear(embed_dim, embed_dim)

        self.ehr_q = nn.Linear(embed_dim, embed_dim)
        self.ehr_k = nn.Linear(embed_dim, embed_dim)
        self.ehr_v = nn.Linear(embed_dim, embed_dim)

        self.img_q = nn.Linear(embed_dim, embed_dim)

        # ==========================================
        # 5. 交叉注意力引擎与降维模块
        # ==========================================
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.2
        )
        
        self.norm_bio_ehr = nn.LayerNorm(embed_dim)
        self.norm_ehr_bio = nn.LayerNorm(embed_dim)
        self.norm_img = nn.LayerNorm(embed_dim)

        self.ab_proj = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # ==========================================
        # 6. 最终决策融合层
        # ==========================================
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, 64), 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(64, 3) 
        )

    def forward(self, x, kg_seq, kg_mask):
        # --- 防御性编程：校验输入的序列长度是否与预设的 max_seq_len 匹配 ---
        if kg_seq.size(1) != self.max_seq_len:
            raise ValueError(f"[Shape Error] Expected graph sequence length {self.max_seq_len}, but got {kg_seq.size(1)}. Please check dataloader output.")
        
        if kg_mask.size(1) != self.max_seq_len:
            raise ValueError(f"[Shape Error] Expected graph mask length {self.max_seq_len}, but got {kg_mask.size(1)}.")

        # ==========================================
        # 阶段 A：输入切片与独立模态编码
        # ==========================================
        ehr_end = self.ehr_dim
        img_end = ehr_end + self.img_dim
        
        ehr = x[:, :ehr_end]                         
        img = x[:, ehr_end:img_end]                  
        bio = x[:, img_end:img_end+self.bio_dim]     

        ehr_feat = self.ehr_encoder(ehr).unsqueeze(1) 
        img_feat = self.img_encoder(img).unsqueeze(1) 
        bio_feat = self.bio_encoder(bio).unsqueeze(1) 

        kg_feat = self.transe_proj(kg_seq)            

        # ==========================================
        # 阶段 B：模态感知知识检索 
        # ==========================================
        ehr_enhanced = ehr_feat 
        
        # 将输入掩码 (1为有效，0为PAD) 转换为 PyTorch 标准 (False为有效，True为忽略)
        # kg_mask 必须转换为 bool 类型
        bool_kg_mask = (kg_mask == 0).bool()
        
        bio_kg_out, _ = self.bio_kg_attn(query=bio_feat, key=kg_feat, value=kg_feat, key_padding_mask=bool_kg_mask)
        img_kg_out, _ = self.img_kg_attn(query=img_feat, key=kg_feat, value=kg_feat, key_padding_mask=bool_kg_mask)

        bio_enhanced = self.norm_bio_kg(bio_feat + bio_kg_out)
        img_enhanced = self.norm_img_kg(img_feat + img_kg_out)
        # ==========================================
        # 阶段 C：底层特征交互
        # ==========================================
        Qa = self.bio_q(bio_enhanced)  
        Ka = self.bio_k(bio_enhanced)
        Va = self.bio_v(bio_enhanced)

        Qb = self.ehr_q(ehr_enhanced)  
        Kb = self.ehr_k(ehr_enhanced)
        Vb = self.ehr_v(ehr_enhanced)

        bio_ehr_attn, _ = self.cross_attention(query=Qa, key=Kb, value=Vb) 
        ehr_bio_attn, _ = self.cross_attention(query=Qb, key=Ka, value=Va) 

        bio_ehr_out = self.norm_bio_ehr(bio_enhanced + bio_ehr_attn)
        ehr_bio_out = self.norm_ehr_bio(ehr_enhanced + ehr_bio_attn)

        ab_combined = torch.cat([bio_ehr_out, ehr_bio_out], dim=1)        
        ab_combined = ab_combined.view(ab_combined.size(0), -1)           
        ab_combined = self.ab_proj(ab_combined).unsqueeze(1)              
        
        # ==========================================
        # 阶段 D：高阶脑影像对齐
        # ==========================================
        Qc = self.img_q(img_enhanced)  
        img_ab_attn, _ = self.cross_attention(query=Qc, key=ab_combined, value=ab_combined) 
        img_out = self.norm_img(img_enhanced + img_ab_attn)               

        # ==========================================
        # 阶段 E：强制隐式知识内化与分类决策
        # ==========================================
        final_combined = torch.cat(
            [
                ab_combined.squeeze(1),   
                img_out.squeeze(1)        
            ],
            dim=1
        ) 
        
        return self.fusion(final_combined)






# 4.22 Perceiver
class PerceiverAttentionBlock(nn.Module):
    """
    Perceiver 核心注意力块。
    使用固定的潜变量（Latent）去查询输入特征，实现信息的压缩与精炼。
    """
    def __init__(self, embed_dim, num_heads, latent_dim=8, dropout=0.2):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, latent_dim, embed_dim))
        
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        b = x.size(0)
        latent = self.latents.expand(b, -1, -1)
        
        # 交叉注意力提取信息
        attn_out, _ = self.cross_attn(query=latent, key=x, value=x, key_padding_mask=mask)
        latent = self.norm1(latent + attn_out)
        
        # 潜变量内部自注意力推理
        attn_out, _ = self.self_attn(query=latent, key=latent, value=latent)
        latent = self.norm2(latent + attn_out)
        
        # 前馈网络非线性映射
        latent = self.norm3(latent + self.ffn(latent))
        
        return latent


class KGMultiModalPerceiver(nn.Module):
    """
    针对 AIBL 数据集定制的：异构知识图谱引导 Perceiver 模型。
    融合了 PPMI 的模态标识符设计与 NACC 的高维统一表征标准。
    """
    def __init__(self, ehr_dim=2, img_dim=64, bio_dim=5, embed_dim=128, 
                 latent_dim_low=16, latent_dim_high=8, num_heads=4, 
                 transe_embed_dim=128, max_seq_len=20):
        super().__init__()
        
        self.ehr_dim = ehr_dim
        self.img_dim = img_dim
        self.bio_dim = bio_dim
        self.max_seq_len = max_seq_len 
        
        # ==========================================
        # 1. 独立模态编码器 (针对 AIBL 小维度特征进行了调整)
        # ==========================================
        # EHR 编码 (输入维度仅为 2: Age, Gender)
        self.ehr_encoder = nn.Sequential(
            nn.Linear(ehr_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
        
        # IMG 编码
        self.img_encoder = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
        
        # BIO 编码 (输入维度为 5: APOE, MMSE, CDR, lm_imm, lm_del)
        self.bio_encoder = nn.Sequential(
            nn.Linear(bio_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )

        # ==========================================
        # 2. 知识图谱精炼模块
        # ==========================================
        self.transe_proj = nn.Sequential(
            nn.Linear(transe_embed_dim, transe_embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(transe_embed_dim, embed_dim)
        )
        
        # ==========================================
        # 3. 模态标识符 (Modal Embedding - 引入 PPMI 核心机制)
        # ==========================================
        # 0:EHR, 1:BIO, 2:IMG, 3:Refined KG
        self.modal_type_embed = nn.Embedding(4, embed_dim)

        # ==========================================
        # 4. Perceiver 核心级联模块
        # ==========================================
        self.kg_refiner = PerceiverAttentionBlock(embed_dim, num_heads, latent_dim=latent_dim_low)
        self.bottom_fusion = PerceiverAttentionBlock(embed_dim, num_heads, latent_dim=latent_dim_low)
        self.high_fusion = PerceiverAttentionBlock(embed_dim, num_heads, latent_dim=latent_dim_high)

        # ==========================================
        # 5. 注意力汇聚与分类器
        # ==========================================
        self.attn_weights = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # 输出维度为 3，对应 AIBL 的三分类任务 (遵循 NACC 标准，无跳跃连接)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128), 
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(128, 3) 
        )

    def forward(self, x, kg_seq, kg_mask):
        # --- 形状安全校验 ---
        if kg_seq.size(1) != self.max_seq_len:
            raise ValueError(f"[Shape Error] Expected graph sequence length {self.max_seq_len}, but got {kg_seq.size(1)}.")

        # A. 特征切片
        ehr_end = self.ehr_dim
        img_end = ehr_end + self.img_dim
        
        ehr = x[:, :ehr_end]                          
        img = x[:, ehr_end:img_end]                  
        bio = x[:, img_end:img_end+self.bio_dim]     

        # B. 独立特征编码
        ehr_feat = self.ehr_encoder(ehr).unsqueeze(1) 
        img_feat = self.img_encoder(img).unsqueeze(1) 
        bio_feat = self.bio_encoder(bio).unsqueeze(1) 

        # C. 知识图谱精炼
        kg_raw = self.transe_proj(kg_seq)

        # 修复 NaN 掩码异常
        kg_mask = kg_mask.clone()
        all_masked = kg_mask.all(dim=1)
        kg_mask[all_masked, 0] = False

        kg_latents = self.kg_refiner(kg_raw, mask=kg_mask)

        # D. 注入模态标识符 (借鉴 PPMI)
        device = x.device
        ehr_feat = ehr_feat + self.modal_type_embed(torch.tensor(0, device=device))
        bio_feat = bio_feat + self.modal_type_embed(torch.tensor(1, device=device))
        img_feat = img_feat + self.modal_type_embed(torch.tensor(2, device=device))
        kg_latents_labeled = kg_latents + self.modal_type_embed(torch.tensor(3, device=device))

        # E. 第一层融合：底层生理特征、临床表型与精炼知识交互
        bottom_seq = torch.cat([ehr_feat, bio_feat, kg_latents_labeled], dim=1)
        bottom_latents = self.bottom_fusion(bottom_seq)

        # F. 第二层融合：底层统一表征与高维脑影像对齐
        high_seq = torch.cat([bottom_latents, img_feat], dim=1)
        final_latents = self.high_fusion(high_seq)

        # G. 注意力汇聚与分类
        weights = self.attn_weights(final_latents) 
        out_feat = torch.sum(final_latents * weights, dim=1) 
        
        return self.classifier(out_feat)







#3.1Perceiver
class KGMultiModalPerceiver_old(nn.Module):
    def __init__(self, embed_dim=16, num_heads=2, num_layers=2, transe_embed_dim=32, num_latents=4):
        super().__init__()
        
        # ---------------- 1. 模态编码器 (保持原逻辑完全不变) ----------------
        self.ehr_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )
        self.img_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )
        self.bio_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )
        self.transe_proj = nn.Linear(transe_embed_dim, embed_dim)

        # ---------------- 2. Perceiver 核心组件 ----------------
        # 潜在张量 (Latents)：作为 Query 读取多模态信息
        self.latents = nn.Parameter(torch.randn(1, num_latents, embed_dim))
        
        # 【新增】位置/模态编码：因为输入序列固定为 6 个元素 (bio, bio_kg, ehr, ehr_kg, img, img_kg)
        # 这让 Attention 机制能够区分当前读取的是哪种模态
        self.pos_embed = nn.Parameter(torch.randn(1, 6, embed_dim))
        
        # 交叉注意力 (Cross-Attention)：Latents (Q) 去读取 Inputs (K, V)
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        # 【新增】交叉注意力的归一化与前馈网络 (FFN)，保证特征提取和梯度收敛
        self.cross_ln_q = nn.LayerNorm(embed_dim)
        self.cross_ln_kv = nn.LayerNorm(embed_dim)
        self.cross_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # 潜在特征自注意力 (Latent Self-Attention)：在 Latents 内部进行深度交互
        # 【新增】使用 ModuleDict 将自注意力的 LN 和 FFN 封装在一起
        self.latent_layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True),
                'ln': nn.LayerNorm(embed_dim),
                'ffn': nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.ReLU(),
                    nn.Linear(embed_dim * 2, embed_dim)
                )
            }) for _ in range(num_layers)
        ])
        
        # ---------------- 3. 输出与融合层 (保持原逻辑完全不变) ----------------
        self.latent_proj = nn.Linear(num_latents * embed_dim, embed_dim * 2)

        # 对齐分类需求的融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )

    def forward(self, x, transe_embed):
        # --- 1. 特征提取与拼接 ---
        ehr = x[:, 0].unsqueeze(1)
        img = x[:, 1].unsqueeze(1)
        bio = x[:, 2].unsqueeze(1)

        ehr_feat = self.ehr_encoder(ehr).unsqueeze(1)
        img_feat = self.img_encoder(img).unsqueeze(1)
        bio_feat = self.bio_encoder(bio).unsqueeze(1)

        transe_feat = self.transe_proj(transe_embed).unsqueeze(1)

        bio_feat = torch.cat([bio_feat, transe_feat], dim=1)
        ehr_feat = torch.cat([ehr_feat, transe_feat], dim=1)
        img_feat = torch.cat([img_feat, transe_feat], dim=1)

        # 拼接后的输入序列形状: [batch, 6, embed_dim]
        inputs_seq = torch.cat([bio_feat, ehr_feat, img_feat], dim=1)
        
        # 【新增】注入模态位置编码，打破 Attention 的置换不变性
        inputs_seq = inputs_seq + self.pos_embed

        batch_size = x.size(0)
        # 扩展 latents 以匹配 batch_size
        latents_batch = self.latents.expand(batch_size, -1, -1)

        # --- 2. Perceiver 交叉注意力 (Read) ---
        # 预归一化 (Pre-LayerNorm) 是 Transformer 架构的最佳实践
        q = self.cross_ln_q(latents_batch)
        kv = self.cross_ln_kv(inputs_seq)
        
        attn_out, _ = self.cross_attention(query=q, key=kv, value=kv)
        # 残差连接 + FFN
        latent_out = latents_batch + attn_out
        latent_out = latent_out + self.cross_ffn(latent_out)

        # --- 3. Perceiver 自注意力 (Process) ---
        for layer in self.latent_layers:
            # 同样使用 Pre-LayerNorm 和残差连接
            qkv = layer['ln'](latent_out)
            attn_out, _ = layer['attn'](query=qkv, key=qkv, value=qkv)
            
            latent_out = latent_out + attn_out
            latent_out = latent_out + layer['ffn'](latent_out)

        # --- 4. 展平与融合 ---
        latent_flat = latent_out.reshape(batch_size, -1)
        modalities_combined = self.latent_proj(latent_flat)

        final_combined = torch.cat(
            [
                modalities_combined,          # [batch, embed_dim * 2]
                transe_feat.squeeze(1)        # [batch, embed_dim]
            ],
            dim=1
        )  # 最终拼接维度为 [batch, embed_dim * 3]
        
        return self.fusion(final_combined)
        


##6.22 AlignMamba
import torch.nn.functional as F
from einops import rearrange, repeat

class SelectiveScan(nn.Module):
    """选择性扫描机制 - Mamba的核心"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        
        # 投影层
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # 参数化 A 矩阵
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.d_state + 1).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
    def forward(self, x, delta, B, C):
        """
        x: [B, L, d_inner]
        delta: [B, L, d_inner]
        B: [B, L, d_state]
        C: [B, L, d_state]
        """
        b, l, d = x.shape
        A = -torch.exp(self.A_log)  # [d_inner, d_state]
        
        y = torch.zeros_like(x)
        h = torch.zeros((b, d, self.d_state), device=x.device)
        
        for i in range(l):
            curr_delta = delta[:, i, :].unsqueeze(-1)
            curr_B = B[:, i, :].unsqueeze(1)
            curr_C = C[:, i, :].unsqueeze(-1)
            
            dA = torch.exp(curr_delta * A.unsqueeze(0))
            dB = curr_delta * curr_B
            
            h = dA * h + dB * x[:, i, :].unsqueeze(-1)
            y[:, i, :] = (h @ curr_C).squeeze(-1)
        
        return y + x * self.D
    

class MambaBlock(nn.Module):
    """标准 Mamba Block"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        
        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 1D 卷积
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # SSM 层
        self.ssm = SelectiveScan(d_model, d_state, d_conv, expand)
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Dropout 和 LayerNorm
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        x: [B, L, D]
        """
        residual = x
        
        # 输入投影并分割
        x_and_res = self.in_proj(x)  # [B, L, 2*d_inner]
        x, res = x_and_res.chunk(2, dim=-1)
        
        # 1D 卷积
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :x.size(-1)]
        x = rearrange(x, 'b d l -> b l d')
        x = F.silu(x)
        
        # SSM 分支
        x_dbl = self.ssm.x_proj(x)
        delta, B, C = x_dbl.split([self.d_inner, self.ssm.d_state, self.ssm.d_state], dim=-1)
        delta = F.softplus(self.ssm.dt_proj(delta))
        
        y = self.ssm(x, delta, B, C)
        y = y * F.silu(res)
        
        # 输出投影
        out = self.out_proj(y)
        out = self.dropout(out)
        
        return self.norm(out + residual)


class CrossModalAlignment(nn.Module):
    """跨模态对齐模块 - AlignMamba的核心"""
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 多头交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 门控融合机制
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # 投影层
        self.proj1 = nn.Linear(d_model, d_model)
        self.proj2 = nn.Linear(d_model, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, modality1, modality2):
        """
        modality1: [B, L1, D] - 第一个模态
        modality2: [B, L2, D] - 第二个模态
        返回对齐后的特征
        """
        # 投影到共同空间
        proj1 = self.proj1(modality1)
        proj2 = self.proj2(modality2)
        
        # 双向交叉注意力
        # modality1 关注 modality2
        attn1, _ = self.cross_attn(proj1, proj2, proj2)
        # modality2 关注 modality1
        attn2, _ = self.cross_attn(proj2, proj1, proj1)
        
        # 门控融合
        gate1 = self.gate(torch.cat([proj1, attn1], dim=-1))
        gate2 = self.gate(torch.cat([proj2, attn2], dim=-1))
        
        aligned1 = gate1 * attn1 + (1 - gate1) * proj1
        aligned2 = gate2 * attn2 + (1 - gate2) * proj2
        
        # 归一化
        aligned1 = self.norm(aligned1)
        aligned2 = self.norm(aligned2)
        
        return aligned1, aligned2


class AlignMambaLayer(nn.Module):
    """AlignMamba 层 - 结合 Mamba 和跨模态对齐"""
    def __init__(self, d_model, d_state=16, num_heads=8, dropout=0.1):
        super().__init__()
        
        # Mamba 块（每个模态独立处理）
        self.mamba1 = MambaBlock(d_model, d_state, expand=2, dropout=dropout)
        self.mamba2 = MambaBlock(d_model, d_state, expand=2, dropout=dropout)
        
        # 跨模态对齐
        self.alignment = CrossModalAlignment(d_model, num_heads, dropout)
        
        # 融合后的 MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, modality1, modality2):
        """
        modality1: [B, L1, D]
        modality2: [B, L2, D]
        """
        # Step 1: 独立的 Mamba 处理
        mamba_out1 = self.mamba1(modality1)
        mamba_out2 = self.mamba2(modality2)
        
        # Step 2: 跨模态对齐
        aligned1, aligned2 = self.alignment(mamba_out1, mamba_out2)
        
        # Step 3: 拼接并融合
        fused = torch.cat([aligned1.mean(dim=1), aligned2.mean(dim=1)], dim=-1)
        fused = self.fusion_mlp(fused)
        
        return self.norm(fused)

class MultiModalAlignMamba(nn.Module):
    def __init__(self, embed_dim=16, d_state=16, num_heads=2, dropout=0.2):
        super().__init__()
        # 模态编码器：将 EHR、MRI（img）、BIO 的原始观测值映射到统一的嵌入空间
        self.ehr_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )
        self.img_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )
        self.bio_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )
        
        # 第一层 AlignMamba：融合 EHR 和 BIO
        self.align_ehr_bio = AlignMambaLayer(
            d_model=embed_dim, 
            d_state=d_state, 
            num_heads=num_heads, 
            dropout=dropout
        )
        
        # 第二层 AlignMamba：将初步融合的结果与 3D-CNN 预处理的 IMG 融合
        self.align_combined_img = AlignMambaLayer(
            d_model=embed_dim, 
            d_state=d_state, 
            num_heads=num_heads, 
            dropout=dropout
        )
        
        # 融合层 (完全保留你原来的结构和 3 分类输出)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        # 输入分解
        ehr = x[:, 0].unsqueeze(1)  # [batch, 1]
        img = x[:, 1].unsqueeze(1)
        bio = x[:, 2].unsqueeze(1)
        
        # 模态编码
        ehr_feat = self.ehr_encoder(ehr).unsqueeze(1)  # [batch, 1, embed_dim]
        img_feat = self.img_encoder(img).unsqueeze(1)
        bio_feat = self.bio_encoder(bio).unsqueeze(1)
        
        # 1. EHR 和 BIO 进行跨模态对齐
        # AlignMambaLayer 输出维度是 [batch, embed_dim]
        ab_combined = self.align_ehr_bio(ehr_feat, bio_feat) 
        
        # 把它增加一个维度，变成 [batch, 1, embed_dim]，方便输入下一层
        ab_combined_seq = ab_combined.unsqueeze(1) 
        
        # 2. 上面的融合特征与 IMG 特征进行跨模态对齐
        # 输出维度是 [batch, embed_dim]
        img_ab_aligned = self.align_combined_img(img_feat, ab_combined_seq) 
        
        # 3. 拼接两个阶段的特征
        # 维度变成 [batch, 2 * embed_dim]，正好符合你原来融合层的输入要求
        final_combined = torch.cat([ab_combined, img_ab_aligned], dim=1) 
        
        # 输入到融合层
        return self.fusion(final_combined)
    


#6.23 CAmera
class CrossAttentionBlock(nn.Module):
    """
    交叉注意力块 - CAMera的核心组件
    实现双向跨模态注意力交互
    """
    def __init__(self, d_model, num_heads=8, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        
        self.d_model = d_model
        
        # 模态1 -> 模态2 的交叉注意力
        self.cross_attn_1to2 = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 模态2 -> 模态1 的交叉注意力
        self.cross_attn_2to1 = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # LayerNorm
        self.norm1_1 = nn.LayerNorm(d_model)
        self.norm1_2 = nn.LayerNorm(d_model)
        self.norm2_1 = nn.LayerNorm(d_model)
        self.norm2_2 = nn.LayerNorm(d_model)
        
        # MLP
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp_1 = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model)
        )
        self.mlp_2 = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, modality1, modality2):
        """
        modality1: [B, L1, D]
        modality2: [B, L2, D]
        """
        # 保存残差
        res1, res2 = modality1, modality2
        
        # Step 1: 交叉注意力
        norm1 = self.norm1_1(modality1)
        norm2 = self.norm1_2(modality2)
        
        attn_1to2, _ = self.cross_attn_1to2(norm1, norm2, norm2)
        attn_2to1, _ = self.cross_attn_2to1(norm2, norm1, norm1)
        
        # 残差连接
        modality1 = modality1 + self.dropout(attn_1to2)
        modality2 = modality2 + self.dropout(attn_2to1)
        
        # Step 2: MLP
        modality1 = modality1 + self.dropout(self.mlp_1(self.norm2_1(modality1)))
        modality2 = modality2 + self.dropout(self.mlp_2(self.norm2_2(modality2)))
        
        return modality1, modality2


class ModalityEncoder(nn.Module):
    """
    单模态编码器 - 使用自注意力提取模态内特征
    """
    def __init__(self, input_dim, d_model, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 自注意力层
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        """
        x: [B, input_dim]
        返回: [B, 1, d_model]
        """
        # 投影并添加序列维度
        x = self.input_proj(x).unsqueeze(1)  # [B, 1, D]
        
        # 自注意力编码
        for layer in self.self_attn_layers:
            x = layer(x)
            
        return x


class AdaptiveFusion(nn.Module):
    """
    自适应融合模块 - 动态调整模态权重
    (保留你原有的定义，以备其他逻辑需要)
    """
    def __init__(self, d_model):
        super().__init__()
        
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, feat1, feat2):
        """
        feat1, feat2: [B, D]
        返回加权融合的特征
        """
        combined = torch.cat([feat1, feat2], dim=-1)
        weights = self.gate_network(combined)  # [B, 2]
        
        fused = weights[:, 0:1] * feat1 + weights[:, 1:2] * feat2
        
        return fused, weights


class MultiModalCAMera(nn.Module):
    """
    适配三模态的 CAMera 模型
    """
    def __init__(
        self,
        input_dim_ehr=1,
        input_dim_img=1,
        input_dim_bio=1,
        num_classes=3,      # 脑部疾病三分类 (AD, Normal, MCI)
        d_model=128,
        num_cross_layers=2, 
        num_self_layers=1,  
        num_heads=8,
        dropout=0.3,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        
        # 1. 三个独立的模态编码器
        self.encoder_ehr = ModalityEncoder(
            input_dim=input_dim_ehr, d_model=d_model,
            num_layers=num_self_layers, num_heads=num_heads, dropout=dropout
        )
        self.encoder_img = ModalityEncoder(
            input_dim=input_dim_img, d_model=d_model,
            num_layers=num_self_layers, num_heads=num_heads, dropout=dropout
        )
        self.encoder_bio = ModalityEncoder(
            input_dim=input_dim_bio, d_model=d_model,
            num_layers=num_self_layers, num_heads=num_heads, dropout=dropout
        )
        
        # 2. EHR 和 BIO 的交叉注意力网络
        self.cross_layers_ehr_bio = nn.ModuleList([
            CrossAttentionBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_cross_layers)
        ])
        
        # 3. 维度调整层 (把拼接后的特征变回 d_model)
        self.ab_proj = nn.Linear(2 * d_model, d_model)
        
        # 4. 融合特征与 IMG 的交叉注意力网络
        self.cross_layers_ab_img = nn.ModuleList([
            CrossAttentionBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_cross_layers)
        ])
        
        # 5. 最终的融合与分类层
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # 第一步：把输入分开
        # 这里的 x 形状是 [batch, 3]
        ehr = x[:, 0].unsqueeze(1)  # 形状: [batch, 1]
        img = x[:, 1].unsqueeze(1)  # 形状: [batch, 1]
        bio = x[:, 2].unsqueeze(1)  # 形状: [batch, 1]
        
        # 第二步：对每个模态进行独立编码
        # 编码后的输出形状都是 [batch, 1, d_model]
        enc_ehr = self.encoder_ehr(ehr)
        enc_img = self.encoder_img(img)
        enc_bio = self.encoder_bio(bio)
        
        # 第三步：EHR 和 BIO 进行交叉注意力交互
        for cross_layer in self.cross_layers_ehr_bio:
            enc_ehr, enc_bio = cross_layer(enc_ehr, enc_bio)
            
        # 第四步：合并 EHR 和 BIO 的特征
        # 把它们拼接起来，形状变成 [batch, 1, 2 * d_model]
        ab_combined = torch.cat([enc_ehr, enc_bio], dim=-1)
        # 把维度调整回 d_model，形状变成 [batch, 1, d_model]
        ab_combined = self.ab_proj(ab_combined)
        
        # 第五步：合并后的特征和 IMG 进行交叉注意力交互
        for cross_layer in self.cross_layers_ab_img:
            ab_combined, enc_img = cross_layer(ab_combined, enc_img)
            
        # 第六步：去掉多余维度并且拼接
        feat_ab = ab_combined.squeeze(1)  # 形状: [batch, d_model]
        feat_img = enc_img.squeeze(1)     # 形状: [batch, d_model]
        final_combined = torch.cat([feat_ab, feat_img], dim=-1)  # 形状: [batch, 2 * d_model]
        
        # 第七步：输入到融合层得到最终的分类结果
        logits = self.fusion(final_combined)
        
        return logits
