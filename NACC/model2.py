import torch
import torch.nn as nn

# ==========================================
# 1. 改良版 3D CNN (只提取特征，不分类)
# ==========================================
class FeatureExtractor3D(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        # 保持和 model1 类似的卷积结构，以便利用预训练权重(如果有的话)
        # 但我们更推荐直接重新训练或微调
        self.features = nn.Sequential(
            # Input: 1 x 64 x 64 x 64
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), # -> 16 x 32 x 32 x 32
            
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), # -> 32 x 16 x 16 x 16
            
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1), # 加一层卷积提取更深层特征
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), # -> 64 x 8 x 8 x 8
        )
        
        # 计算扁平化后的维度: 64通道 * 8 * 8 * 8 = 32768
        self.flatten_dim = 64 * 8 * 8 * 8
        
        # 投影到目标特征维度 (如 128)
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_dim), # 输出特征向量
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x # 返回 [Batch, 128]

# ==========================================
# 2. 改良版 EHR Net (处理表格数据)
# ==========================================
class EHRFeatureNet(nn.Module):
    def __init__(self, input_dim=24, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim), # 输出特征向量
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x) # 返回 [Batch, 32]

# ==========================================
# 3. 真正平衡的三模态融合模型
# ==========================================
class AdvancedTriModalFusion(nn.Module):
    def __init__(self, 
                 ehr_input_dim=32, 
                 mri_input_dim=128, 
                 kg_input_dim=128, 
                 common_dim=64, 
                 num_classes=3):
        super().__init__()
        
        # --- 1. 模态对齐 (Projection Layers) ---
        # 将不同维度的输入映射到同一个 common_dim
        self.ehr_proj = nn.Linear(ehr_input_dim, common_dim)
        self.mri_proj = nn.Linear(mri_input_dim, common_dim)
        self.kg_proj = nn.Linear(kg_input_dim, common_dim)
        
        # --- 2. 融合机制 (Concat + MLP) ---
        # 这种方法比复杂的 Cross-Attention 在小样本下更稳定
        self.fusion_net = nn.Sequential(
            nn.Linear(common_dim * 3, 128), # 拼接后是 64*3=192
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # --- 3. 分类头 ---
        self.classifier = nn.Linear(64, num_classes)
        
    def forward(self, ehr, mri, kg):
        # ehr: [B, 32], mri: [B, 128], kg: [B, 128]
        
        # 1. 对齐
        f_ehr = self.ehr_proj(ehr) # -> [B, 64]
        f_mri = self.mri_proj(mri) # -> [B, 64]
        f_kg  = self.kg_proj(kg)   # -> [B, 64]
        
        # 2. 拼接
        combined = torch.cat([f_ehr, f_mri, f_kg], dim=1) # -> [B, 192]
        
        # 3. 融合与分类
        feat = self.fusion_net(combined)
        logits = self.classifier(feat)
        
        return logits