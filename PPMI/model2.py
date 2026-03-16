import torch
import torch.nn as nn
import numpy as np

# =========================================================================
# 1. 3D CNN: 用于从原始 MRI 影像 (64x64x64) 中提取特征
# =========================================================================
class ImageEncoder3D(nn.Module):
    def __init__(self, output_dim=64):
        super().__init__()
        # 特征提取部分：3层 3D 卷积
        self.features = nn.Sequential(
            # Input: [Batch, 1, 64, 64, 64]
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2), # -> [16, 32, 32, 32]
            
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2), # -> [32, 16, 16, 16]
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2), # -> [64, 8, 8, 8]
        )
        
        # 展平并映射到统一维度
        # Flatten size: 64 * 8 * 8 * 8 = 32768
        self.fc = nn.Sequential(
            nn.Linear(32768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim) # 输出维度 (例如 64)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        return self.fc(x)

# =========================================================================
# 2. 核心融合模型: 基于 Transformer (原版)
# =========================================================================
class MedicalFusionTransformer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, num_layers=3, kg_dim=128, ehr_dim=18):
        super().__init__()
        
        self.img_encoder = ImageEncoder3D(output_dim=embed_dim)
        
        self.ehr_encoder = nn.Sequential(
            nn.Linear(ehr_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )
        
        self.kg_encoder = nn.Sequential(
            nn.Linear(kg_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, 32), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4) 
        )

    def forward(self, img, ehr, kg):
        feat_img = self.img_encoder(img).unsqueeze(1)
        feat_ehr = self.ehr_encoder(ehr).unsqueeze(1)
        feat_kg  = self.kg_encoder(kg).unsqueeze(1)
        
        seq = torch.cat([feat_img, feat_ehr, feat_kg], dim=1)
        out_seq = self.transformer(seq)
        
        out_flat = out_seq.view(out_seq.size(0), -1) 
        logits = self.classifier(out_flat)
        return logits

# =========================================================================
# 3. 核心融合模型: 基于 Mamba (更低显存，线性复杂度)
# =========================================================================
class MedicalFusionMamba(nn.Module):
    def __init__(self, embed_dim=64, num_layers=2, kg_dim=128, ehr_dim=18, d_state=16, d_conv=4, expand=2):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError:
            raise ImportError("请先安装 Mamba: pip install mamba-ssm causal-conv1d>=1.2.0")

        # --- 模态编码器 (复用) ---
        self.img_encoder = ImageEncoder3D(output_dim=embed_dim)
        
        self.ehr_encoder = nn.Sequential(
            nn.Linear(ehr_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )
        
        self.kg_encoder = nn.Sequential(
            nn.Linear(kg_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        
        # --- 融合层 (Mamba Stack) ---
        # 相比 Transformer，Mamba 没有 Attention 矩阵，显存占用更小
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=embed_dim, # Model dimension d_model
                d_state=d_state,   # SSM state expansion factor
                d_conv=d_conv,     # Local convolution width
                expand=expand,     # Block expansion factor
            ) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # --- 分类头 ---
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, img, ehr, kg):
        # 1. 独立编码
        feat_img = self.img_encoder(img).unsqueeze(1)
        feat_ehr = self.ehr_encoder(ehr).unsqueeze(1)
        feat_kg  = self.kg_encoder(kg).unsqueeze(1)
        
        # 2. 构建序列 [Image, EHR, KG]
        seq = torch.cat([feat_img, feat_ehr, feat_kg], dim=1) # [B, 3, D]
        
        # 3. Mamba 序列建模
        for layer in self.mamba_layers:
            seq = layer(seq)
        
        seq = self.norm(seq)
        
        # 4. 展平并分类
        out_flat = seq.view(seq.size(0), -1)
        logits = self.classifier(out_flat)
        
        return logits