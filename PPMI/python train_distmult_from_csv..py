#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_distmult_from_csv_PD.py
-----------------------------
1) 把 PD1.csv / control.csv / prodromal.csv / swedd.csv 转成 <h, r, t> 三元组
2) 切分 train / valid / test (80 / 10 / 10)
3) 用 PyKEEN 训练 DistMult (embedding_dim = 32)
4) 输出 entity2id_distmult_PD.txt / relation2id_distmult_PD.txt / entity_embeddings_distmult_PD.npy
"""

import os, re, random, numpy as np, pandas as pd, torch
from pathlib import Path
from tqdm import tqdm
import pykeen.pipeline as pk_pipeline

# ========= 基础配置 ========= #
CSV_FILES = {
    "PD":        "PD1.csv",
    "Control":   "control.csv",
    "Prodromal": "prodromal.csv",
    "SWEDD":     "swedd.csv",
}

EMBED_DIM = 32
SEED      = 2025

PREFIX          = "distmult_PD"                    # ← 所有输出文件将带这个前缀
TRIPLE_ALL      = f"triples_{PREFIX}.tsv"
TRIPLE_TRAIN    = f"train_{PREFIX}.tsv"
TRIPLE_VALID    = f"valid_{PREFIX}.tsv"
TRIPLE_TEST     = f"test_{PREFIX}.tsv"
ENTITY2ID_PATH  = f"entity2id_{PREFIX}.txt"
REL2ID_PATH     = f"relation2id_{PREFIX}.txt"
EMB_PATH        = f"entity_embeddings_{PREFIX}.npy"

VALID_FRAC = 0.10
TEST_FRAC  = 0.10

# ========= 固定随机种子 ========= #
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ========= 工具函数 ========= #
def sanitize(text: str) -> str:
    """去除空白并替换为下划线，防止 TSV 解析冲突"""
    return re.sub(r"\s+", "_", str(text).strip())

def age_bin(age) -> str:
    """将年龄离散到 10 岁区间"""
    try:
        a = float(age)
    except Exception:
        return "age_unknown"
    lo = int(a // 10 * 10)
    return f"age_{lo}_{lo + 9}"

def make_triples(row: pd.Series, group_name: str):
    """根据一行数据生成三元组列表"""
    h = sanitize(row["Subject"]) if "Subject" in row else sanitize(row["Image Data ID"])
    triples = [(h, "belongs_to_group", group_name)]

    # 性别、年龄、访视期
    if "Sex" in row and pd.notna(row["Sex"]):
        triples.append((h, "has_sex", sanitize(row["Sex"])))
    if "Age" in row and pd.notna(row["Age"]):
        triples.append((h, "has_age_bin", age_bin(row["Age"])))
    if "Visit" in row and pd.notna(row["Visit"]):
        triples.append((h, "visit_phase", sanitize(row["Visit"])))

    # 其他列（从第 8 列开始），值不为 0/空 时保留
    for col in row.index[8:]:
        v = row[col]
        if pd.isna(v) or str(v).strip() in ("", "0", "0.0"):
            continue
        triples.append((h, sanitize(f"has_{col}"), sanitize(v)))
    return triples

# ========= Step 1：生成三元组 ========= #
all_triples = []
print(">> Generating triples …")
for group, csv_path in CSV_FILES.items():
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"{csv_path} not found.")
    df = pd.read_csv(csv_path)
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"Processing {csv_path}"):
        all_triples.extend(make_triples(r, group))

print(f">> Total triples generated: {len(all_triples):,}")

with open(TRIPLE_ALL, "w", encoding="utf-8") as fw:
    for h, r, t in all_triples:
        fw.write(f"{h}\t{r}\t{t}\n")
print(f">> All triples dumped to {TRIPLE_ALL}")

# ========= Step 2：随机切分 ========= #
random.shuffle(all_triples)
n_total = len(all_triples)
n_test  = int(n_total * TEST_FRAC)
n_valid = int(n_total * VALID_FRAC)

test_triples  = all_triples[:n_test]
valid_triples = all_triples[n_test:n_test + n_valid]
train_triples = all_triples[n_test + n_valid:]

def dump(path, triples):
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")

dump(TRIPLE_TRAIN, train_triples)
dump(TRIPLE_VALID, valid_triples)
dump(TRIPLE_TEST,  test_triples)

print(f">> Split  Train:{len(train_triples):,}  Valid:{len(valid_triples):,}  Test:{len(test_triples):,}")

# ========= Step 3：PyKEEN 训练 DistMult ========= #
print(">> Starting PyKEEN training (DistMult)…")
pipeline_result = pk_pipeline.pipeline(
    model="DistMult",
    training=TRIPLE_TRAIN,
    validation=TRIPLE_VALID,
    testing=TRIPLE_TEST,
    model_kwargs=dict(embedding_dim=EMBED_DIM),
    negative_sampler="basic",
    training_kwargs=dict(num_epochs=200, batch_size=1024),
    stopper="early",
    stopper_kwargs=dict(frequency=10, patience=10, relative_delta=0.002),
    evaluator_kwargs=dict(filtered=True),
    random_seed=SEED,
    device="cpu",         # 如有 GPU 改 "cuda"
    training_loop="SLCWA",
)

print(">> Training finished!   Best MRR:", pipeline_result.get_metric("mean_reciprocal_rank"))

# ========= Step 4：保存实体 / 关系映射和嵌入 ========= #
tf_train = pipeline_result.training     # TriplesFactory
model    = pipeline_result.model        # DistMult

ent2id = tf_train.entity_to_id
rel2id = tf_train.relation_to_id
emb_matrix = model.entity_representations[0]().cpu().detach().numpy()

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
print("✅ All done – embeddings ready for your Notebook.")
