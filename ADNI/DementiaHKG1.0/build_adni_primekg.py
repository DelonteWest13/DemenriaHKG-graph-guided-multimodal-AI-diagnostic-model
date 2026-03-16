import pandas as pd
import numpy as np
import os
import json
import csv
import re
from tqdm import tqdm

# ================= 1. 配置区 =================
CSV_FILES = ['AD.csv', 'mci.csv', 'normal.csv']
PRIMEKG_PATH = "primekg_ad_only.csv"

# 输出文件
OUTPUT_TRIPLETS = "adni_knowledge_triplets.csv"
OUTPUT_E2ID = "adni_kg_entity2id.json"
OUTPUT_R2ID = "adni_kg_relation2id.json"

# ================= 2. 映射与分箱配置 =================

# 2.1 必须排除的列 (防泄露)
EXCLUDE_COLS = {
    'path', 'filename', 'PTID',
    'NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD', 'OTHER',
    'mmse', 'cdr', 'cdrSum'  # 金标准评分，排除
}

# 2.2 PrimeKG 症状映射
SYMPTOM_MAP = {
    "npiq_DEL": "PrimeKG:Delusions",
    "npiq_HALL": "PrimeKG:Hallucinations",
    "npiq_AGIT": "PrimeKG:Agitation",
    "npiq_DEPD": "PrimeKG:Depressivity",
    "npiq_ANX": "PrimeKG:Anxiety",
    "npiq_ELAT": "PrimeKG:Conspicuously happy disposition",
    "npiq_APA": "PrimeKG:Apathy",
    "npiq_DISN": "PrimeKG:Disinhibition",
    "npiq_IRR": "PrimeKG:Irritability",
    "npiq_MOT": "PrimeKG:Restlessness",
}

# 2.3 本地强特征前缀 (必须保留)
IMPORTANT_PREFIXES = [
    "Tesla", "faq_", "his_", "trail", "lm_", "boston",
    "animal", "vege", "digit", "gds", "moca"
]

# 基础属性
base_col_map = {
    "age": "Concept:Age",
    "gender": "Concept:Sex",
    "education": "Concept:Education",
    "race": "Concept:Race"
}


# ================= 3. 数值分箱函数 (防止过拟合的核心) =================
def discretize_val(col_name, val):
    """
    将连续数值离散化，确保相似的数值连到同一个节点。
    """
    try:
        v = float(val)
    except:
        return str(val).strip().lower()

    col_lower = col_name.lower()

    # 1. MRI / 脑容量 (通常是大数值或特定比率)
    if "tesla" in col_lower or "brain" in col_lower:
        # 策略: 如果是大数 (>100)，按 50 分箱；如果是小数，保留1位
        if v > 100:
            return str(int(v // 50 * 50))  # e.g. 1234 -> 1200
        else:
            return f"{v:.1f}"

    # 2. 年龄
    if "age" in col_lower:
        return str(int(v // 5 * 5))  # 每5岁一档

    # 3. 连线测试 (时间秒数)
    if "trail" in col_lower:
        return str(int(v // 10 * 10))  # 每10秒一档

    # 4. 评分量表 (FAQ, GDS, MoCA) -> 直接取整
    if any(x in col_lower for x in ["faq", "gds", "moca", "digit", "boston", "animal"]):
        return str(int(v))

    # 默认保留2位小数
    return f"{v:.2f}"


# ================= 4. 处理逻辑 =================
def load_primekg():
    print(f"🔹 读取 PrimeKG: {PRIMEKG_PATH} ...")
    if not os.path.exists(PRIMEKG_PATH): return []
    triplets = []
    try:
        df = pd.read_csv(PRIMEKG_PATH, low_memory=False)
        cols = df.columns
        x_col = next((c for c in cols if 'x_name' in c or 'head' in c), 'x_name')
        y_col = next((c for c in cols if 'y_name' in c or 'tail' in c), 'y_name')
        r_col = next((c for c in cols if 'relation' in c), 'relation')
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing PrimeKG"):
            triplets.append((f"PrimeKG:{row[x_col]}", str(row[r_col]), f"PrimeKG:{row[y_col]}"))
    except Exception as e:
        print(f"❌ 读取失败: {e}")
    return triplets


def process_adni_files():
    local_triplets = []
    ptid_pattern = re.compile(r"(\d+_S_\d+)")

    for file_name in CSV_FILES:
        if not os.path.exists(file_name): continue
        print(f"    处理 ADNI 文件: {file_name} ...")
        df = pd.read_csv(file_name)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Parsing {file_name}"):
            # --- ID 提取 ---
            if 'filename' in row and pd.notna(row['filename']):
                raw = str(row['filename'])
                match = ptid_pattern.search(raw)
                pid = f"Patient:{match.group(1)}" if match else f"Patient:{raw}"
            else:
                continue

            # --- 基础属性 ---
            for col, rel in base_col_map.items():
                if col in row and pd.notna(row[col]):
                    val_bin = discretize_val(col, row[col])
                    local_triplets.append((pid, "has_attribute", f"{rel}:{val_bin}"))

            # --- APOE ---
            if 'apoe' in row and pd.notna(row['apoe']):
                try:
                    aval = int(float(row['apoe']))
                    if aval > 0:
                        gnode = f"Gene:APOE_e4_Copies:{aval}"
                        local_triplets.append((pid, "has_gene_risk", gnode))
                        local_triplets.append((gnode, "risk_factor_for", "PrimeKG:Alzheimer disease"))
                except:
                    pass

            # --- 强特征与 PrimeKG 映射 ---
            for col in df.columns:
                if col in EXCLUDE_COLS or col in base_col_map or col == 'apoe': continue
                val = row[col]
                if pd.isna(val) or str(val) in ['', 'nan', '0', '0.0']: continue

                # A. 症状 (PrimeKG)
                mapped_pkg = None
                for k, v in SYMPTOM_MAP.items():
                    if k in col: mapped_pkg = v; break

                if mapped_pkg:
                    snode = f"Symptom:{col}"
                    # 直连 Shortcut (重要!)
                    local_triplets.append((pid, "exhibits", mapped_pkg))
                    # 细节路径
                    local_triplets.append((pid, "exhibits_detail", snode))
                    local_triplets.append((snode, "severity", f"Level:{int(float(val))}"))
                    continue

                # B. 本地强特征 (MRI, FAQ...)
                is_imp = False
                for prefix in IMPORTANT_PREFIXES:
                    if col.startswith(prefix): is_imp = True; break

                if is_imp:
                    # ★ 使用分箱后的值 ★
                    val_bin = discretize_val(col, val)
                    fnode = f"Feature:{col}_{val_bin}"
                    local_triplets.append((pid, "has_clinical_measure", fnode))

    return local_triplets


def main():
    kt = load_primekg()
    at = process_adni_files()
    all_t = kt + at

    df = pd.DataFrame(all_t, columns=['head', 'relation', 'tail'])
    print(f"原始: {len(df)}")
    df.drop_duplicates(inplace=True)
    print(f"去重后: {len(df)}")

    df.to_csv(OUTPUT_TRIPLETS, index=False)

    ents = sorted(list(set(df['head']) | set(df['tail'])))
    rels = sorted(list(set(df['relation'])))

    with open(OUTPUT_E2ID, 'w') as f: json.dump({e: i for i, e in enumerate(ents)}, f)
    with open(OUTPUT_R2ID, 'w') as f: json.dump({r: i for i, r in enumerate(rels)}, f)
    print("✅ 图谱构建完成！")


if __name__ == "__main__":
    main()