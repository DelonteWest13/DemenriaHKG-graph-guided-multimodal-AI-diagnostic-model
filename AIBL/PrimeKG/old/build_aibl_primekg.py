import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm

# ================= 1. 配置区 =================
# AIBL 本地数据文件 (文件名大小写需与实际文件完全一致)
CSV_FILES = ['AD.csv', 'MCI.csv', 'NC.csv']

# 指向第一步生成的通用生物底座
PRIMEKG_PATH = "primekg_ad_only.csv"

# 输出文件
OUTPUT_TRIPLETS = "aibl_knowledge_triplets.csv"
OUTPUT_E2ID = "aibl_kg_entity2id.json"
OUTPUT_R2ID = "aibl_kg_relation2id.json"

# ================= 2. 映射逻辑定义 =================

# 2.1 基础属性映射 (移除了 CSV 中不存在的 race 和 education)
base_col_map = {
    "age": "Concept:Age",
    "gender": "Concept:Sex"
}


def load_primekg():
    """读取筛选后的 PrimeKG 生物学子图"""
    print(f"🔹 正在读取 PrimeKG 子图: {PRIMEKG_PATH} ...")
    if not os.path.exists(PRIMEKG_PATH):
        print(f"❌ 错误：找不到 {PRIMEKG_PATH}，请先运行 filter_primekg.py")
        return []

    triplets = []
    try:
        df = pd.read_csv(PRIMEKG_PATH, low_memory=False)
        cols = df.columns
        x_col = next((c for c in cols if 'x_name' in c or 'head' in c), 'x_name')
        y_col = next((c for c in cols if 'y_name' in c or 'tail' in c), 'y_name')
        r_col = next((c for c in cols if 'relation' in c), 'relation')

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing PrimeKG"):
            h = "PrimeKG:" + str(row[x_col])
            t = "PrimeKG:" + str(row[y_col])
            r = str(row[r_col])
            triplets.append((h, r, t))

        print(f" ✅ 已加载医学知识: {len(triplets)} 条")
    except Exception as e:
        print(f" ❌ 读取 PrimeKG 失败: {e}")
    return triplets


def process_aibl_files():
    """处理 AIBL CSV 文件"""
    local_triplets = []
    patient_ids = set()

    for file_name in CSV_FILES:
        if not os.path.exists(file_name):
            print(f"⚠️ 跳过缺失文件: {file_name}")
            continue

        print(f"    正在处理 AIBL 文件: {file_name} ...")
        df = pd.read_csv(file_name)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Parsing {file_name}"):
            # 1. 构建病人 ID (保留完整文件名，包括 .nii 后缀)
            if 'filename' in row and pd.notna(row['filename']):
                s_id = str(row['filename']).strip()
            else:
                continue

            patient_id = f"Patient:{s_id}"
            patient_ids.add(patient_id)

            # ===============================================================
            # ✅ 2. 基础属性 (Age, Gender)
            # ===============================================================
            for col, relation_name in base_col_map.items():
                if col in row and pd.notna(row[col]):
                    val = str(row[col])
                    target_node = f"{relation_name}:{val}"
                    local_triplets.append((patient_id, "has_attribute", target_node))

            # ===============================================================
            # ✅ 3. 影像采集参数 (Tesla) - 新增
            # ===============================================================
            if 'Tesla' in row and pd.notna(row['Tesla']):
                try:
                    tesla_val = float(row['Tesla'])
                    # 格式化为 1.5T 或 3.0T
                    tesla_node = f"Concept:MRI_Strength:{tesla_val}T"
                    local_triplets.append((patient_id, "scanned_with", tesla_node))
                except:
                    pass

            # ===============================================================
            # ✅ 4. 核心基因 (APOE)
            # ===============================================================
            if 'apoe' in row and pd.notna(row['apoe']):
                try:
                    apoe_val = int(float(row['apoe']))
                    # 如果 apoe > 0 (通常指携带 E4 等位基因)
                    if apoe_val > 0:
                        gene_node = "Gene:APOE_e4_Carrier"
                        local_triplets.append((patient_id, "has_gene_risk", gene_node))
                        # 补充基因本身的背景知识
                        local_triplets.append((gene_node, "risk_factor_for", "PrimeKG:Alzheimer disease"))
                        local_triplets.append((gene_node, "is_variant_of", "PrimeKG:APOE"))
                except:
                    pass

            # ===============================================================
            # ✅ 5. 认知评分 (MMSE, CDR, Logical Memory)
            # ===============================================================

            # --- MMSE ---
            if 'mmse' in row and pd.notna(row['mmse']):
                try:
                    mmse = int(float(row['mmse']))
                    mmse_node = f"Test:MMSE:{mmse}"
                    local_triplets.append((patient_id, "has_mmse_score", mmse_node))
                    # 简单的医学规则连接 (可选)
                    if mmse < 24:
                        local_triplets.append((mmse_node, "indicates", "PrimeKG:Dementia"))
                except:
                    pass

            # --- CDR ---
            if 'cdr' in row and pd.notna(row['cdr']):
                try:
                    cdr = float(row['cdr'])
                    cdr_node = f"Test:CDR:{cdr}"
                    local_triplets.append((patient_id, "has_cdr_score", cdr_node))
                except:
                    pass

            # --- Logical Memory Immediate (lm_imm) - 新增 ---
            if 'lm_imm' in row and pd.notna(row['lm_imm']):
                try:
                    val = int(float(row['lm_imm']))
                    node = f"Test:LM_Immediate:{val}"
                    local_triplets.append((patient_id, "has_memory_score", node))
                except:
                    pass

            # --- Logical Memory Delayed (lm_del) - 新增 ---
            if 'lm_del' in row and pd.notna(row['lm_del']):
                try:
                    val = int(float(row['lm_del']))
                    node = f"Test:LM_Delayed:{val}"
                    local_triplets.append((patient_id, "has_memory_score", node))
                except:
                    pass

    print(f" ✅ AIBL 临床数据处理完成，生成: {len(local_triplets)} 条关系")
    return local_triplets


def main():
    # 1. 加载
    kg_triplets = load_primekg()
    aibl_triplets = process_aibl_files()

    if not aibl_triplets:
        print("❌ 严重警告：未生成任何 AIBL 临床数据！请检查 CSV_FILES 列表是否与你的文件名一致。")
        return

    # 2. 合并
    all_triplets = kg_triplets + aibl_triplets
    print(f"\n📊 原始生成总数量: {len(all_triplets)}")

    # 3. 保存与去重
    df_out = pd.DataFrame(all_triplets, columns=['head', 'relation', 'tail'])

    before_dedup = len(df_out)
    df_out.drop_duplicates(inplace=True)
    after_dedup = len(df_out)

    print(f"✂️  执行去重操作: {before_dedup} -> {after_dedup} (移除 {before_dedup - after_dedup})")

    df_out.to_csv(OUTPUT_TRIPLETS, index=False)
    print(f"💾 最终 AIBL 异构图谱已保存至: {OUTPUT_TRIPLETS}")

    # 4. 生成 ID 映射
    print("🔹 正在生成 Entity/Relation ID 映射表...")
    entities = sorted(list(set(df_out['head']) | set(df_out['tail'])))
    relations = sorted(list(set(df_out['relation'])))

    ent2id = {e: i for i, e in enumerate(entities)}
    rel2id = {r: i for i, r in enumerate(relations)}

    with open(OUTPUT_E2ID, 'w') as f: json.dump(ent2id, f)
    with open(OUTPUT_R2ID, 'w') as f: json.dump(rel2id, f)

    print(f"   实体总数: {len(entities)}")
    print(f"   关系总数: {len(relations)}")
    print(f"   映射表已保存: {OUTPUT_E2ID}, {OUTPUT_R2ID}")

    # 简单验证
    print("\n🔍 验证环节: 检查是否有 NC 病人")
    nc_count = sum(1 for e in entities if "Patient:" in e and "S236160" in e)  # 使用你提供的 NC 样本 ID 检查
    print(f"   包含示例 NC 病人 (S236160)? {'✅ 是' if nc_count > 0 else '❌ 否'}")


if __name__ == "__main__":
    main()