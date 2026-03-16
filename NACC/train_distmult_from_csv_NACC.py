#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_distmult_from_csv_NACC.py
-------------------------------
[修正版 - 解决 PyKEEN 验证集实体未知导致的崩溃]
1) 读取 CSV，按 Patient ID 内存切分 Train/Valid/Test
2) 丢弃 Valid/Test 组的病人 (完全不让 KG 看到)
3) 仅对 Train 组病人的三元组进行内部 9:1 切分 (用于 Early Stopping)
4) 训练并保存 Embeddings
"""

import os, re, random, numpy as np, pandas as pd, torch
from pathlib import Path
from tqdm import tqdm
import pykeen.pipeline as pk_pipeline
from sklearn.model_selection import train_test_split

# ========= 基础配置 ========= #
CSV_FILES = {
    "Normal": "NACC_normal.csv",
    "MCI": "NACC_mci.csv",
    "AD": "NACC_ad.csv",
}

EMBED_DIM = 32
SEED = 2025

PREFIX = "distmult_NACC"
TRIPLE_TRAIN = f"train_{PREFIX}.tsv"
TRIPLE_VALID = f"valid_{PREFIX}.tsv"
TRIPLE_TEST = f"test_{PREFIX}.tsv"

ENTITY2ID_PATH = f"entity2id_{PREFIX}.txt"
REL2ID_PATH = f"relation2id_{PREFIX}.txt"
EMB_PATH = f"entity_embeddings_{PREFIX}.npy"

# ID 划分比例 (这里指留给下游任务的比例)
# KG 训练只用剩下的 Train 部分
VALID_SIZE_FOR_DOWNSTREAM = 0.10
TEST_SIZE_FOR_DOWNSTREAM = 0.10

# ========= 固定随机种子 ========= #
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ========= 工具函数 ========= #
def sanitize(text: str) -> str:
    return re.sub(r"\s+", "_", str(text).strip())


def age_bin(age) -> str:
    try:
        a = float(age)
    except Exception:
        return "age_unknown"
    lo = int(a // 10 * 10)
    return f"age_{lo}_{lo + 9}"


def make_triples(row: pd.Series):
    h = sanitize(row['ID'])  # 修正：直接提取真实的患者ID列
    triples = []

    # 属性处理
    if "gender" in row and pd.notna(row["gender"]):
        triples.append((h, "has_gender", sanitize(row["gender"])))
    elif "Sex" in row and pd.notna(row["Sex"]):
        triples.append((h, "has_sex", sanitize(row["Sex"])))

    if "age" in row and pd.notna(row["age"]):
        triples.append((h, "has_age_bin", age_bin(row["age"])))
    elif "Age" in row and pd.notna(row["Age"]):
        triples.append((h, "has_age_bin", age_bin(row["Age"])))

    # 从第 27 列开始遍历特征
    for col in row.index[27:]:
        v = row[col]
        if pd.isna(v) or str(v).strip() in ("", "0", "0.0", "nan"):
            continue
        # 修正：将特征名与数值拼接，防止不同特征的相同分值指向同一个实体
        triples.append((h, f"has_{sanitize(col)}", sanitize(f"{col}_{v}")))

    return triples


# ========= Step 1: 读取数据 & ID 划分 ========= #
print(">> Loading Data & Splitting IDs...")
all_records = []

for group, csv_path in CSV_FILES.items():
    if not Path(csv_path).exists():
        print(f"Warning: {csv_path} not found.")
        continue
    df = pd.read_csv(csv_path)
    for idx, row in df.iterrows():
        subj_id = str(row['ID']) # 修正：直接提取真实的患者ID列
        all_records.append({
            "id": subj_id,
            "row": row
        })

# 提取唯一 ID
unique_ids = list(set(r['id'] for r in all_records))
unique_ids.sort()

# 1. 这里划分出的 test_ids 和 val_ids 是留给下游任务用的
# 2. KG 训练绝对不能看这些 ID，否则就是泄露
train_ids, test_ids = train_test_split(unique_ids, test_size=TEST_SIZE_FOR_DOWNSTREAM, random_state=SEED)
train_ids, val_ids = train_test_split(train_ids, test_size=VALID_SIZE_FOR_DOWNSTREAM / (1 - TEST_SIZE_FOR_DOWNSTREAM),
                                      random_state=SEED)

train_id_set = set(train_ids)

print(f">> Downstream ID Split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
print(f">> Note: Only 'Train' patients will be used for KG construction.")

# ========= Step 2: 生成训练用的三元组 ========= #
all_train_triples = []

print(">> Generating triples from TRAIN patients only...")
for rec in tqdm(all_records):
    pid = rec['id']
    # 严防泄露：只处理训练集病人
    if pid in train_id_set:
        triples = make_triples(rec['row'])
        all_train_triples.extend(triples)

print(f">> Total triples from train patients: {len(all_train_triples):,}")

# ========= Step 3: 内部切分 (解决 PyKEEN 崩溃的关键) ========= #
# 我们不能用 val_ids 做验证，因为那些实体在 all_train_triples 里不存在。
# 解决方法：把 all_train_triples 切分 90% 训练，10% 验证。
# 这样验证集里的实体（病人ID）必然在训练集中出现过。

random.shuffle(all_train_triples)
num_val = int(len(all_train_triples) * 0.1)

kg_train_triples = all_train_triples[num_val:]
kg_val_triples = all_train_triples[:num_val]

print(f">> PyKEEN Internal Train: {len(kg_train_triples):,}")
print(f">> PyKEEN Internal Valid: {len(kg_val_triples):,}")


# 写入文件
def dump_triples(path, triples):
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


dump_triples(TRIPLE_TRAIN, kg_train_triples)
dump_triples(TRIPLE_VALID, kg_val_triples)
dump_triples(TRIPLE_TEST, kg_val_triples)  # 占位，不影响

# ========= Step 4: PyKEEN 训练 ========= #
print(">> Starting PyKEEN training (DistMult)...")

# 检测是否有 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f">> Using device: {device}")

pipeline_result = pk_pipeline.pipeline(
    model="DistMult",
    training=TRIPLE_TRAIN,
    validation=TRIPLE_VALID,
    testing=TRIPLE_TEST,
    model_kwargs=dict(embedding_dim=EMBED_DIM),
    negative_sampler="basic",
    # 如果显存/内存不足，可以调小 batch_size，例如 512 或 1024
    training_kwargs=dict(num_epochs=100, batch_size=1024),
    stopper="early",
    stopper_kwargs=dict(frequency=5, patience=5, relative_delta=0.002),
    evaluator_kwargs=dict(filtered=True),
    random_seed=SEED,
    device=device,
    training_loop="SLCWA",
)

metric = pipeline_result.get_metric("mean_reciprocal_rank")
print(f">> Training finished! Best MRR: {metric}")

# ========= Step 5: 保存结果 ========= #
tf_train = pipeline_result.training
model = pipeline_result.model

ent2id = tf_train.entity_to_id
rel2id = tf_train.relation_to_id
# 移动到 CPU 并转为 numpy
emb_matrix = model.entity_representations[0](indices=None).cpu().detach().numpy()

with open(ENTITY2ID_PATH, "w", encoding="utf-8") as f:
    for ent, idx in ent2id.items():
        f.write(f"{ent}\t{idx}\n")
print(f">> entity2id saved to {ENTITY2ID_PATH}")

with open(REL2ID_PATH, "w", encoding="utf-8") as f:
    for rel, idx in rel2id.items():
        f.write(f"{rel}\t{idx}\n")
print(f">> relation2id saved to {REL2ID_PATH}")

np.save(EMB_PATH, emb_matrix)
print(f">> entity embeddings saved to {EMB_PATH}")
print("✅ Done.")