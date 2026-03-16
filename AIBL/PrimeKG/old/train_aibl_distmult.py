import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import json
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# ================= ⚡ 配置区 (已修改为 AIBL) =================
# 1. 输入文件 (对应 build_aibl_primekg.py 的输出)
TRIPLETS_FILE = 'aibl_knowledge_triplets.csv'    # <--- 修改点 1
ENTITY2ID_FILE = 'aibl_kg_entity2id.json'        # <--- 修改点 2
RELATION2ID_FILE = 'aibl_kg_relation2id.json'    # <--- 修改点 3

# 2. 输出文件 (训练好的 Embeddings)
OUTPUT_EMBED = 'aibl_kg_embeddings.npy'          # <--- 修改点 4

# 3. 训练超参数 (保持与 ADNI 一致，方便横向对比)
EMBED_DIM = 128
NUM_EPOCHS = 200
BATCH_SIZE = 512
LR = 0.005
TRAIN_RATIO = 0.9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 [AIBL] 训练设备: {DEVICE}")


# ================= 🛠️ 数据加载类 (通用) =================
class KGDataset(Dataset):
    def __init__(self, triplets_file, entity2id, relation2id):
        print(f"    正在读取图谱文件: {triplets_file} ...")

        try:
            df = pd.read_csv(triplets_file)
        except Exception as e:
            print(f"❌ 读取 CSV 失败: {e}")
            self.triplets = []
            return

        print(f"    ✅ 成功加载原始数据: {len(df)} 行")

        self.triplets = []
        skipped = 0

        # 将字符串转换为 ID
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing Data"):
            try:
                h_token = str(row['head']).strip()
                r_token = str(row['relation']).strip()
                t_token = str(row['tail']).strip()

                h = entity2id.get(h_token)
                r = relation2id.get(r_token)
                t = entity2id.get(t_token)

                if h is not None and r is not None and t is not None:
                    self.triplets.append((h, r, t))
                else:
                    skipped += 1
            except Exception:
                skipped += 1
                continue

        self.triplets = torch.LongTensor(self.triplets)

        print(f"    📊 最终有效三元组: {len(self.triplets)}")
        if skipped > 0:
            print(f"    ⚠️ 跳过了 {skipped} 条数据 (正常现象)")

        if len(self.triplets) == 0:
            raise ValueError("❌ 错误：没有生成任何有效数据！请检查 entity2id 是否匹配。")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


# ================= 🧠 DistMult 模型 (通用) =================
class DistMult(nn.Module):
    def __init__(self, num_entities, num_relations, embed_dim):
        super(DistMult, self).__init__()
        self.num_entities = num_entities
        self.ent_emb = nn.Embedding(num_entities, embed_dim)
        self.rel_emb = nn.Embedding(num_relations, embed_dim)

        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        self.criterion = nn.MarginRankingLoss(margin=1.0)

    def forward(self, h, r, t):
        h_e = self.ent_emb(h)
        r_e = self.rel_emb(r)
        t_e = self.ent_emb(t)
        score = torch.sum(h_e * r_e * t_e, dim=1)
        return score

    def calculate_loss(self, h, r, t):
        batch_size = h.size(0)
        neg_t = torch.randint(0, self.num_entities, (batch_size,), device=h.device)
        pos_score = self.forward(h, r, t)
        neg_score = self.forward(h, r, neg_t)
        target = torch.ones(batch_size, device=h.device)
        loss = self.criterion(pos_score, neg_score, target)
        return loss


# ================= 🏃 主训练循环 =================
def train():
    # 1. 检查文件
    if not os.path.exists(ENTITY2ID_FILE):
        print(f"❌ 找不到 {ENTITY2ID_FILE}，请先运行 build_aibl_primekg.py")
        return

    # 2. 加载映射
    print("📥 正在加载 ID 映射表...")
    with open(ENTITY2ID_FILE, 'r') as f:
        entity2id = json.load(f)
    with open(RELATION2ID_FILE, 'r') as f:
        relation2id = json.load(f)

    num_ents = len(entity2id)
    num_rels = len(relation2id)
    print(f"    实体总数: {num_ents} | 关系总数: {num_rels}")

    # 3. 准备数据
    dataset = KGDataset(TRIPLETS_FILE, entity2id, relation2id)

    train_size = int(TRAIN_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # 4. 初始化模型
    model = DistMult(num_ents, num_rels, EMBED_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\n🔥 [AIBL] 开始训练 (共 {NUM_EPOCHS} 轮)...")

    # 5. 循环
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", leave=False)

        for batch in progress:
            batch = batch.to(DEVICE)
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            optimizer.zero_grad()
            loss = model.calculate_loss(h, r, t)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)

        if (epoch) % 1 == 0:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(DEVICE)
                    h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
                    loss = model.calculate_loss(h, r, t)
                    test_loss += loss.item()
            avg_test = test_loss / len(test_loader)
            print(f"Epoch {epoch + 1:03d} | 📉 Train Loss: {avg_loss:.4f} | 🔍 Test Loss: {avg_test:.4f}")

    # 6. 保存
    print("\n" + "=" * 40)
    print(f"💾 训练完成！正在保存 AIBL Embeddings 至: {OUTPUT_EMBED}")

    embeddings = model.ent_emb.weight.detach().cpu().numpy()
    np.save(OUTPUT_EMBED, embeddings)

    print(f"✅ 保存成功！矩阵形状: {embeddings.shape}")
    print("=" * 40)


if __name__ == "__main__":
    train()