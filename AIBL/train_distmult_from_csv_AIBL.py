#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_distmult_from_csv_AIBL_fixed.py
-------------------------------------
[修正说明 - 防止数据泄露]
1. 移除了 'belongs_to_group' 三元组，不再将诊断标签(NC/MCI/AD)放入图谱。
   知识图谱现在只包含客观临床特征（年龄、性别、MMSE、APOE等）。
2. 生成的 Embedding 将只代表病人的临床特征分布，不再包含答案。

功能：
1) 读取 NC.csv / MCI.csv / AD.csv
2) 转成 <h, r, t> 三元组 (仅特征)
3) 用 PyKEEN 训练 DistMult
4) 输出 entity2id / relation2id / entity_embeddings
"""

import os
import re
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import pykeen.pipeline as pk_pipeline

# ========= 1. 基础配置 ========= #
CSV_FILES = {
    "NC": "NC.csv",
    "MCI": "MCI.csv",
    "AD": "AD.csv",
}

EMBED_DIM = 32
SEED = 2025

# 输出文件命名（建议加上 _fixed 后缀以示区别）
PREFIX = "distmult_AIBL_fixed"
TRIPLE_ALL = f"triples_{PREFIX}.tsv"
TRIPLE_TRAIN = f"train_{PREFIX}.tsv"
TRIPLE_VALID = f"valid_{PREFIX}.tsv"
TRIPLE_TEST = f"test_{PREFIX}.tsv"
ENTITY2ID_PATH = f"entity2id_{PREFIX}.txt"
REL2ID_PATH = f"relation2id_{PREFIX}.txt"
EMB_PATH = f"entity_embeddings_{PREFIX}.npy"

VALID_FRAC = 0.10
TEST_FRAC = 0.10

# 需要跳过的列（不作为KG特征）
SKIP_COLS = {
    "path", "filename",  # 标识符/路径
    "visit", "age", "gender",  # 单独处理的特征
    "NC", "MCI", "DE", "COG",  # 诊断/认知标签 -> 必须跳过防止泄露
    "AD", "PD", "FTD", "VD", "DLB", "PDD", "ADD", "OTHER",  # 其他诊断标签 -> 必须跳过
}

# 明确指定需要保留的特征列
# 确保这些列只包含临床数据，不包含任何暗示最终诊断的标签
FEATURE_COLS = [
    "apoe", "mmse", "cdr", "lm_imm", "lm_del", "Tesla",
]

# ========= 2. 固定随机种子 ========= #
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ========= 3. 工具函数 ========= #
def sanitize(text: str) -> str:
    """去除空白并替换为下划线，防止 TSV 解析冲突"""
    return re.sub(r"\s+", "_", str(text).strip())


def age_bin(age) -> str:
    """将年龄离散到 10 岁区间，减少稀疏性"""
    try:
        a = float(age)
    except Exception:
        return "age_unknown"
    lo = int(a // 10 * 10)
    return f"age_{lo}_{lo + 9}"


def make_triples(row: pd.Series, group_name: str):
    """
    生成三元组逻辑：
    - 实体头(h): Subject ID (使用 filename)
    - 关系(r): has_age, has_sex, has_mmse 等
    - 实体尾(t): 具体的值

    【重要修改】：不再生成 (ID, belongs_to_group, Label) 三元组
    """
    # 1) 受试者 ID：优先用 filename 作为唯一标识
    if "filename" in row and pd.notna(row["filename"]):
        h_raw = row["filename"]
    else:
        # 兜底：如果没 filename，尝试用第 2 列 (row.iloc[1])，视具体CSV结构而定
        h_raw = row.iloc[1]

    h = sanitize(h_raw)
    triples = []

    # --- [修正点]：绝对不要把 group_name (NC/AD) 放进去！---
    # triples.append((h, "belongs_to_group", group_name))  <-- 已删除

    # 2) 基本属性：性别 / 年龄 / 访视
    if "gender" in row and pd.notna(row["gender"]):
        val = sanitize(row["gender"])
        if val not in ("", "0", "nan"):
            triples.append((h, "has_sex", val))

    if "age" in row and pd.notna(row["age"]):
        val = age_bin(row["age"])
        triples.append((h, "has_age_bin", val))

    if "visit" in row and pd.notna(row["visit"]):
        val = sanitize(row["visit"])
        if val not in ("", "nan"):
            triples.append((h, "visit_phase", val))

    # 3) 其余表型/量表特征 (APOE, MMSE, CDR 等)
    for col in row.index:
        # 跳过不需要的列
        if col in SKIP_COLS:
            continue
        # 只处理在 FEATURE_COLS 里的列
        if FEATURE_COLS and col not in FEATURE_COLS:
            continue

        v = row[col]
        if pd.isna(v):
            continue

        v_str = str(v).strip()
        # 跳过空值或无意义的0值
        if v_str in ("", "nan", "None"):
            continue

        # 数值类型的特征处理（可选）：
        # 如果某些特征是连续数值（如Tesla），直接作为实体会导致图太稀疏
        # 简单起见，这里直接当作实体，如果数值太多，建议先进行分箱(Binning)

        triples.append((h, f"has_{sanitize(col)}", sanitize(v_str)))

    return triples


# ========= Step 1：生成三元组 ========= #
all_triples = []
print(">> Generating triples for AIBL (Clean Version - No Labels) …")

for group, csv_path in CSV_FILES.items():
    if not Path(csv_path).exists():
        print(f"Warning: {csv_path} not found, skipping.")
        continue

    df = pd.read_csv(csv_path)
    print(f"   Processing {group}: {len(df)} rows")

    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"   Parsing {csv_path}"):
        # 传入 group 仅用于 debug 或日志，make_triples 内部不再将其加入图谱
        triples = make_triples(r, group)
        all_triples.extend(triples)

print(f">> Total triples generated: {len(all_triples):,}")

# 简单的去重，防止重复行导致的问题
all_triples = list(set(all_triples))
print(f">> Unique triples: {len(all_triples):,}")

with open(TRIPLE_ALL, "w", encoding="utf-8") as fw:
    for h, r, t in all_triples:
        fw.write(f"{h}\t{r}\t{t}\n")
print(f">> All triples dumped to {TRIPLE_ALL}")

# ========= Step 2：随机切分 ========= #
# 对于下游分类任务，这里的切分主要是为了训练 KG Embedding
# 即使 Test Set 的病人在 KG Train Set 里出现了属性边 (has_age, 70) 也是允许的（Transductive Setting）
# 只要不出现 (ID, is, AD) 这种标签边即可。
random.shuffle(all_triples)
n_total = len(all_triples)
n_test = int(n_total * TEST_FRAC)
n_valid = int(n_total * VALID_FRAC)

test_triples = all_triples[:n_test]
valid_triples = all_triples[n_test:n_test + n_valid]
train_triples = all_triples[n_test + n_valid:]


def dump(path, triples):
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


dump(TRIPLE_TRAIN, train_triples)
dump(TRIPLE_VALID, valid_triples)
dump(TRIPLE_TEST, test_triples)

print(f">> Split  Train:{len(train_triples):,}  Valid:{len(valid_triples):,}  Test:{len(test_triples):,}")

# ========= Step 3：PyKEEN 训练 DistMult ========= #
# 检查是否有 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f">> Starting PyKEEN training (DistMult) on {device}…")

pipeline_result = pk_pipeline.pipeline(
    model="DistMult",
    training=TRIPLE_TRAIN,
    validation=TRIPLE_VALID,
    testing=TRIPLE_TEST,
    model_kwargs=dict(embedding_dim=EMBED_DIM),
    # 训练参数微调
    negative_sampler="basic",
    training_kwargs=dict(num_epochs=100, batch_size=2048),  # 稍微减小epoch，增大batch加快速度
    stopper="early",
    stopper_kwargs=dict(frequency=5, patience=5, relative_delta=0.002),
    evaluator_kwargs=dict(filtered=True),
    random_seed=SEED,
    device=device,
    training_loop="SLCWA",
)

# 获取指标 (MRR) - 仅代表图补全能力，不代表下游分类能力
metric = pipeline_result.get_metric("mean_reciprocal_rank")
print(f">> Training finished!   Best MRR: {metric:.4f}")

# ========= Step 4：保存实体 / 关系映射和嵌入 ========= #
tf_train = pipeline_result.training  # TriplesFactory
model = pipeline_result.model  # DistMult

ent2id = tf_train.entity_to_id
rel2id = tf_train.relation_to_id

# 取出实体嵌入矩阵 (detach 转 numpy)
emb_matrix = model.entity_representations[0]().detach().cpu().numpy()

# 保存 ID 映射
with open(ENTITY2ID_PATH, "w", encoding="utf-8") as f:
    for ent, idx in ent2id.items():
        f.write(f"{ent}\t{idx}\n")
print(f">> entity2id saved to {ENTITY2ID_PATH}")

with open(REL2ID_PATH, "w", encoding="utf-8") as f:
    for rel, idx in rel2id.items():
        f.write(f"{rel}\t{idx}\n")
print(f">> relation2id saved to {REL2ID_PATH}")

# 保存 Embedding 矩阵
np.save(EMB_PATH, emb_matrix)
print(f">> entity embeddings saved to {EMB_PATH}  shape={emb_matrix.shape}")
print("✅ All done. Please use the *_fixed files in your multimodal training pipeline.")