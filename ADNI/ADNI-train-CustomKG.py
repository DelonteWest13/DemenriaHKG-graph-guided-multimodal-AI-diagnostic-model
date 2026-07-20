#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, re, random, numpy as np, pandas as pd, torch
from pathlib import Path
from tqdm import tqdm
import sys

# ========= 1. 基础配置 ========= #
INPUT_CSV = "ADNI诊断.csv"
PREFIX = "distmult_ADNI" # 中间生成文件前缀保持不变

EMBED_DIM = 128
SEED = 2025
FEATURE_START_IDX = 2

# 我们稍后在下一环节处理强相关量表的剔除，这里先保留你的原设置
EXCLUDE_COLS = {
    'path', 'filename', 'PTID', 
    'NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD', 'OTHER', 
    'apoe' 
}

TRIPLE_TRAIN = f"train_{PREFIX}.tsv"
TRIPLE_VALID = f"valid_{PREFIX}.tsv"
TRIPLE_TEST = f"test_{PREFIX}.tsv"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ========= 2. 辅助函数 ========= #
def sanitize(text: str) -> str:
    return re.sub(r"\s+", "_", str(text).strip())

def extract_ptid_from_filename(filename_str):
    match = re.search(r"(\d+_S_\d+)", str(filename_str))
    if match:
        return match.group(1)
    return str(filename_str)

def age_bin(age):
    try:
        a = float(age)
    except:
        return "age_unknown"
    lo = int(a // 10 * 10)
    return f"age_{lo}_{lo + 9}"

def get_triples_for_patient(row: pd.Series):
    """
    生成纯特征三元组，彻底防止数据泄露
    """
    h = sanitize(row['PTID'])
    feature_triples = []

    if "gender" in row and pd.notna(row["gender"]):
        feature_triples.append((h, "has_gender", sanitize(row["gender"])))
    if "age" in row and pd.notna(row["age"]):
        feature_triples.append((h, "has_age_bin", age_bin(row["age"])))

    for col in row.index[FEATURE_START_IDX:]:
        if col in EXCLUDE_COLS or col.lower() in ['age', 'gender', 'ptid']:
            continue

        v = row[col]
        if pd.isna(v) or str(v).strip() in ("", "nan"):
            continue

        col_name = sanitize(col)
        val_str = sanitize(v)
        feature_triples.append((h, f"has_{col_name}", val_str))

    return feature_triples

# ========= 3. 读取数据与构建全局三元组 ========= #
print(f">> Loading data from {INPUT_CSV} ...")
if not Path(INPUT_CSV).exists():
    print(f"Error: File not found {INPUT_CSV}")
    sys.exit(1)

df = pd.read_csv(INPUT_CSV)

if 'filename' not in df.columns:
    raise ValueError(f"{INPUT_CSV} 中缺少 'filename' 列，无法提取ID")

df['PTID'] = df['filename'].apply(extract_ptid_from_filename)

all_triples = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="   Parsing Data"):
    feats = get_triples_for_patient(row)
    all_triples.extend(feats)

# ========= 4. 随机切分三元组 (无监督链路预测切分) ========= #
random.shuffle(all_triples)

n_total = len(all_triples)
n_test = int(n_total * 0.10)
n_valid = int(n_total * 0.10)

final_test = all_triples[:n_test]
final_valid = all_triples[n_test:n_test + n_valid]
final_train = all_triples[n_test + n_valid:]

print(f"\n>> Triples Stats:")
print(f"   Train Triples: {len(final_train):,}")
print(f"   Valid Triples: {len(final_valid):,}")
print(f"   Test  Triples: {len(final_test):,}")

# ========= 5. 保存中间文件 ========= #
def save_tsv(path, triples):
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples: f.write(f"{h}\t{r}\t{t}\n")

save_tsv(TRIPLE_TRAIN, final_train)
save_tsv(TRIPLE_VALID, final_valid)
save_tsv(TRIPLE_TEST, final_test)
print("\n>> Data generation complete. Ready for PyKeen.")

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