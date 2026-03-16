#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_distmult_from_csv_ADNI.py
-------------------------------
修正版：
1. 自动从 filename 列提取 Patient ID (如 109_S_1157)。
2. 解决数据泄露：按病人ID切分，且对测试集病人隐藏 'belongs_to_group' 标签。
3. 自动排除 'AD', 'MCI' 等标签列进入特征三元组。
"""

import os, re, random, numpy as np, pandas as pd, torch
from pathlib import Path
from tqdm import tqdm
import pykeen.pipeline as pk_pipeline
import sys

# ========= 1. 基础配置 ========= #
CSV_FILES = {
    "Normal": "normal.csv",
    "MCI": "mci.csv",
    "AD": "AD.csv",
}

# 沿用原文件名配置，去掉 _final
PREFIX = "distmult_ADNI"

# 训练参数
EMBED_DIM = 128
SEED = 2025

# 特征配置
# 从第几列开始遍历？(0:path, 1:filename, 2:age...) -> 从 2 开始
FEATURE_START_IDX = 2

# !!! 必须排除的列 !!!
# 这些列要么是元数据，要么是直接泄露答案的标签，绝对不能作为特征三元组输入
EXCLUDE_COLS = {
    'path', 'filename', 'PTID',  # 元数据
    'NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD', 'OTHER',  # 诊断标签
    'apoe'  # 遗传信息通常也作为预测目标或强特征，如果你想作为输入特征，请从这里移除 'apoe'
}

TRIPLE_TRAIN = f"train_{PREFIX}.tsv"
TRIPLE_VALID = f"valid_{PREFIX}.tsv"
TRIPLE_TEST = f"test_{PREFIX}.tsv"
ENTITY2ID_PATH = f"entity2id_{PREFIX}.txt"
REL2ID_PATH = f"relation2id_{PREFIX}.txt"
EMB_PATH = f"entity_embeddings_{PREFIX}.npy"

VALID_FRAC = 0.10
TEST_FRAC = 0.10

# 固定随机种子
random.seed(SEED);
np.random.seed(SEED);
torch.manual_seed(SEED)


# ========= 2. 辅助函数 ========= #
def sanitize(text: str) -> str:
    """清理字符串，移除空格等"""
    return re.sub(r"\s+", "_", str(text).strip())


def extract_ptid_from_filename(filename_str):
    """
    从 ADNI 文件名中提取病人ID。
    例如: 'ADNI_109_S_1157_MR_...' -> '109_S_1157'
    """
    # 正则匹配: 3位数字 + _S_ + 4位数字
    match = re.search(r"(\d+_S_\d+)", str(filename_str))
    if match:
        return match.group(1)
    # 如果匹配不到，尝试匹配以此开头的
    return str(filename_str)


def age_bin(age):
    try:
        a = float(age)
    except:
        return "age_unknown"
    lo = int(a // 10 * 10)
    return f"age_{lo}_{lo + 9}"


def get_triples_for_patient(row: pd.Series, group: str):
    """
    生成三元组。
    feature_triples: 给模型看的信息 (Age, Gender, MMSE...)
    label_triples:   要预测的答案 (belongs_to_group)
    """
    # 此时 row 中已经有了我们在循环中生成的 'PTID' 列
    h = sanitize(row['PTID'])

    feature_triples = []
    label_triples = []

    # --- A. 构建预测目标 (Label) ---
    label_triples.append((h, "belongs_to_group", group))

    # --- B. 构建特征 (Features) ---
    # 1. 显式处理性别和年龄 (如果列名存在)
    if "gender" in row and pd.notna(row["gender"]):
        feature_triples.append((h, "has_gender", sanitize(row["gender"])))
    if "age" in row and pd.notna(row["age"]):
        feature_triples.append((h, "has_age_bin", age_bin(row["age"])))

    # 2. 自动遍历剩余数值/量表特征
    # 从索引 FEATURE_START_IDX 开始，跳过 EXCLUDE_COLS 中的列
    for col in row.index[FEATURE_START_IDX:]:
        # 如果该列在排除列表中，或者是 gender/age (已处理)，则跳过
        if col in EXCLUDE_COLS or col.lower() in ['age', 'gender', 'ptid']:
            continue

        v = row[col]
        # 跳过空值
        if pd.isna(v) or str(v).strip() in ("", "nan"):
            continue

        col_name = sanitize(col)
        val_str = sanitize(v)
        feature_triples.append((h, f"has_{col_name}", val_str))

    return h, feature_triples, label_triples


# ========= 3. 读取数据与ID提取 ========= #
print(">> Loading data and extracting Patient IDs ...")
patient_data = []  # List of dicts

for grp, csv_path in CSV_FILES.items():
    if not Path(csv_path).exists():
        print(f"Error: File not found {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    # --- 关键步骤：提取 PTID ---
    if 'filename' not in df.columns:
        raise ValueError(f"{csv_path} 中缺少 'filename' 列，无法提取ID")

    df['PTID'] = df['filename'].apply(extract_ptid_from_filename)

    print(f"   Processed {csv_path}: Found {len(df)} rows.")
    if len(df) > 0:
        print(f"   Sample IDs: {df['PTID'].head(3).tolist()}")

    # 生成三元组
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"   Parsing {grp}"):
        pid, feats, labs = get_triples_for_patient(row, grp)
        patient_data.append({
            "id": pid,
            "feats": feats,
            "labs": labs
        })

# ========= 4. 按病人切分数据集 ========= #
# 获取所有唯一的病人ID
unique_patients = list(set(p["id"] for p in patient_data))
random.shuffle(unique_patients)

n_total = len(unique_patients)
# 安全检查
if n_total < 10:
    print(f"\n⚠️ 严重警告: 提取到的唯一病人数仅为 {n_total}！")
    print("这通常意味着提取逻辑有误，所有行被映射到了同一个ID。")
    print("请检查 `extract_ptid_from_filename` 函数。")
    sys.exit(1)

n_test = int(n_total * TEST_FRAC)
n_valid = int(n_total * VALID_FRAC)
n_train = n_total - n_test - n_valid

test_ids = set(unique_patients[:n_test])
valid_ids = set(unique_patients[n_test:n_test + n_valid])
train_ids = set(unique_patients[n_test + n_valid:])

print(f"\n>> Split Info:")
print(f"   Total Subjects: {n_total}")
print(f"   Train Subjects: {len(train_ids)}")
print(f"   Valid Subjects: {len(valid_ids)}")
print(f"   Test  Subjects: {len(test_ids)}")

# ========= 5. 构建 Masked Datasets (防泄露) ========= #
final_train = []
final_valid = []
final_test = []

for p in patient_data:
    pid = p["id"]

    # 所有人: 特征都放进训练集 (Inductive Context)
    final_train.extend(p["feats"])

    if pid in train_ids:
        # 训练集病人: 标签放进训练集
        final_train.extend(p["labs"])
    elif pid in valid_ids:
        # 验证集病人: 标签放进验证集
        final_valid.extend(p["labs"])
    elif pid in test_ids:
        # 测试集病人: 标签放进测试集
        final_test.extend(p["labs"])

print(f"\n>> Triples Stats:")
print(f"   Train Triples: {len(final_train):,}")
print(f"   Valid Triples: {len(final_valid):,}")
print(f"   Test  Triples: {len(final_test):,}")


# 保存
def save_tsv(path, triples):
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples: f.write(f"{h}\t{r}\t{t}\n")


save_tsv(TRIPLE_TRAIN, final_train)
save_tsv(TRIPLE_VALID, final_valid)
save_tsv(TRIPLE_TEST, final_test)

# ========= 6. 训练模型 ========= #
print("\n>> Starting DistMult training ...")

# 自动检测设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Using device: {device}")

result = pk_pipeline.pipeline(
    model="DistMult",
    training=TRIPLE_TRAIN,
    validation=TRIPLE_VALID,
    testing=TRIPLE_TEST,
    model_kwargs=dict(embedding_dim=EMBED_DIM),
    training_kwargs=dict(num_epochs=100, batch_size=2048),
    stopper="early",
    stopper_kwargs=dict(frequency=5, patience=10, relative_delta=0.002),
    evaluator_kwargs=dict(filtered=True),
    negative_sampler="basic",
    random_seed=SEED,
    device=device,
    training_loop="SLCWA",
)

# ========= 7. 结果与保存 ========= #
print("\n>> Training Finished.")
metrics = result.metric_results.to_dict()

# 尝试打印 MRR
try:
    # PyKeen >= 1.9.0 路径
    mrr = metrics['test']['both']['realistic']['inverse_harmonic_mean_rank']
    print(f"   Test MRR: {mrr:.4f}")
except:
    print("   Metrics available:", metrics.keys())

# 保存 Embedding
model = result.model
entity_embs = model.entity_representations[0]().cpu().detach().numpy()
np.save(EMB_PATH, entity_embs)

# 保存 ID 映射
with open(ENTITY2ID_PATH, "w", encoding="utf-8") as f:
    for e, i in result.training.entity_to_id.items():
        f.write(f"{e}\t{i}\n")

with open(REL2ID_PATH, "w", encoding="utf-8") as f:
    for r, i in result.training.relation_to_id.items():
        f.write(f"{r}\t{i}\n")

print(f"\n✅ All Done. Embeddings saved to {EMB_PATH}")