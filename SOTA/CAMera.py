import torch
import torch.nn as nn
import torch.nn.functional as F


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
        
        # Step 1: 交叉注意力 - 模态1关注模态2
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
        # 计算门控权重
        combined = torch.cat([feat1, feat2], dim=-1)
        weights = self.gate_network(combined)  # [B, 2]
        
        # 加权融合
        fused = weights[:, 0:1] * feat1 + weights[:, 1:2] * feat2
        
        return fused, weights


class CAMera(nn.Module):
    """
    CAMera: Cross-Attention Multi-modal Encoder
    
    专为双模态生物医学数据设计：
    - Modality 1: 基因表达数据 (View)
    - Modality 2: GRN构建的张量数据 (Voxel)
    
    核心特点：
    1. 双向交叉注意力实现模态间深度交互
    2. 自适应融合机制动态调整模态贡献
    3. 多层堆叠逐步优化跨模态表示
    """
    def __init__(
        self,
        input_dim1,          # 第一个模态维度 (X1_shape)
        input_dim2,          # 第二个模态维度 (input_D - X1_shape)
        num_classes=2,
        d_model=128,
        num_cross_layers=2,
        num_self_layers=1,
        num_heads=8,
        dropout=0.3,
        **kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 模态特定编码器
        self.encoder1 = ModalityEncoder(
            input_dim=input_dim1,
            d_model=d_model,
            num_layers=num_self_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.encoder2 = ModalityEncoder(
            input_dim=input_dim2,
            d_model=d_model,
            num_layers=num_self_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 交叉注意力层
        self.cross_layers = nn.ModuleList([
            CrossAttentionBlock(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_cross_layers)
        ])
        
        # 自适应融合
        self.fusion = AdaptiveFusion(d_model)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 辅助损失头（可选，用于增强训练）
        self.aux_classifier1 = nn.Linear(d_model, num_classes)
        self.aux_classifier2 = nn.Linear(d_model, num_classes)
        
        # 初始化权重
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
    
    def forward_dual(self, modality1, modality2):
        """
        双模态前向传播
        
        modality1: [B, input_dim1] - 基因表达数据
        modality2: [B, input_dim2] - GRN张量数据
        
        返回: logits
        """
        # Step 1: 独立的模态编码
        enc1 = self.encoder1(modality1)  # [B, 1, D]
        enc2 = self.encoder2(modality2)  # [B, 1, D]
        
        # Step 2: 多层交叉注意力交互
        for cross_layer in self.cross_layers:
            enc1, enc2 = cross_layer(enc1, enc2)
        
        # Step 3: 提取全局表示（取序列的第一个token）
        feat1 = enc1.squeeze(1)  # [B, D]
        feat2 = enc2.squeeze(1)  # [B, D]
        
        # Step 4: 自适应融合
        fused_feat, fusion_weights = self.fusion(feat1, feat2)
        
        # Step 5: 主分类头
        logits = self.classifier(fused_feat)
        
        # 训练时返回辅助输出
        if self.training:
            aux_logits1 = self.aux_classifier1(feat1)
            aux_logits2 = self.aux_classifier2(feat2)
            return logits, aux_logits1, aux_logits2, fusion_weights
        
        return logits
    
    def forward(self, x):
        """
        兼容单输入接口（需要外部切分）
        实际使用时建议使用 forward_dual
        """
        # 这个方法主要用于兼容性，实际应该在train.py中处理切分
        raise NotImplementedError("请使用 forward_dual(modality1, modality2)")