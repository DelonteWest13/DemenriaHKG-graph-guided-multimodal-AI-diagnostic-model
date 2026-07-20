import torch
import torch.nn as nn
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


class AlignMamba(nn.Module):
    """
    AlignMamba - 基于 Mamba 的多模态对齐分类器
    
    专为你的双模态数据设计：
    - Modality 1: 基因表达数据 (View)
    - Modality 2: GRN 构建的张量数据 (Voxel)
    """
    def __init__(
        self, 
        input_dim1,      # 第一个模态维度 (X1_shape)
        input_dim2,      # 第二个模态维度 (input_D - X1_shape)
        num_classes=2,
        d_model=128,
        d_state=16,
        num_layers=2,
        num_heads=8,
        dropout=0.3,
        **kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 模态特定的嵌入投影
        self.embed1 = nn.Sequential(
            nn.Linear(input_dim1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.embed2 = nn.Sequential(
            nn.Linear(input_dim2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 多层 AlignMamba
        self.layers = nn.ModuleList([
            AlignMambaLayer(d_model, d_state, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        """
        x: [B, input_D] - 拼接后的双模态输入
        需要在外部按 X1_shape 切分
        """
        # 这里假设输入已经是切分好的两个模态
        # 实际使用时需要在 train.py 中处理切分
        pass
    
    def forward_dual(self, modality1, modality2):
        """
        双模态前向传播
        
        modality1: [B, input_dim1] - 基因表达数据
        modality2: [B, input_dim2] - GRN 张量数据
        """
        # Step 1: 嵌入投影
        emb1 = self.embed1(modality1).unsqueeze(1)  # [B, 1, D]
        emb2 = self.embed2(modality2).unsqueeze(1)  # [B, 1, D]
        
        # Step 2: 多层 AlignMamba 处理
        fused = None
        for layer in self.layers:
            fused = layer(emb1, emb2)  # [B, D]
            # 为下一层准备输入（如果需要可以传递更多信息）
            emb1 = fused.unsqueeze(1)
            emb2 = fused.unsqueeze(1)
        
        # Step 3: 分类
        logits = self.classifier(fused)
        
        return logits