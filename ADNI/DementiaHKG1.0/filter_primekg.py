import pandas as pd
import numpy as np

# ================= 配置区 =================
RAW_KG_PATH = "kg.csv"
OUTPUT_PATH = "primekg_ad_only.csv"  # 输出文件名改为 AD

# ★★★ 核心修改：使用验证成功的“关键词军团” ★★★
# 逻辑：只要包含列表中的任意一个词，就认为是 AD 相关
KEYWORDS = [
    "Alzheimer",
    "Dementia",
    "Memory",
    "Cognitive",
    "Amyloid",
    "Tau",
    "Neurofibrillary"
]

# 必须保留的非药物节点类型 (保持纯净生物学背景)
VALID_TYPES = [
    'disease',
    'gene/protein',
    'effect/phenotype',
    'pathway',
    'biological_process',
    'anatomy'
]


def main():
    print(f"🔹 1. 正在加载原始 PrimeKG ({RAW_KG_PATH}) ...")
    # low_memory=False 防止大文件读取警告
    df = pd.read_csv(RAW_KG_PATH, low_memory=False)

    # ================= 核心筛选逻辑 =================
    print(f"🔹 2. 正在根据关键词军团进行生物学筛选: {KEYWORDS} ...")

    # A. 名字匹配 (升级版)：使用正则表达式进行“或”匹配
    # 'Alzheimer|Dementia|Amyloid...'
    pattern = '|'.join(KEYWORDS)

    mask_name = (df['x_name'].str.contains(pattern, case=False, na=False)) | \
                (df['y_name'].str.contains(pattern, case=False, na=False))

    # B. 类型清洗：剔除 Drug, Indication 等 (只保留 VALID_TYPES)
    mask_type = (df['x_type'].isin(VALID_TYPES)) & (df['y_type'].isin(VALID_TYPES))

    # C. 关系清洗：双重保险，剔除药物关系
    mask_rel = ~df['relation'].isin(['indication', 'contraindication', 'drug_drug', 'off-label use'])

    # 应用筛选
    subset = df[mask_name & mask_type & mask_rel].copy()

    if len(subset) == 0:
        print("❌ 错误：没有找到任何符合条件的数据！请检查关键词。")
        return

    print("\n" + "=" * 40)
    print(f"✅ 初步筛选 AD 相关知识: {len(subset)} 条")
    print("=" * 40)

    # ================= 📊 统计打印区 =================
    print("\n[详细分类统计]")
    rel_counts = subset['relation'].value_counts()
    for rel, count in rel_counts.items():
        note = ""
        if rel == 'disease_protein':
            note = "(疾病-基因)"
        elif rel == 'disease_phenotype_positive':
            note = "(典型症状)"
        elif rel == 'pathway_protein':
            note = "(通路-基因)"
        print(f"  - {rel:<30} : {count:>5} 条 {note}")

    # ================= 补充一步 PPI (基因互作) =================
    # AD 涉及很多蛋白（Amyloid, Tau, APOE），我们需要知道它们之间有没有相互作用
    print("\n🔹 3. 正在补充基因内部互作关系 (PPI) 以完善图结构...")

    # 1. 拿到刚才筛选出的所有基因
    related_genes = set(subset[subset['y_type'] == 'gene/protein']['y_name']) | \
                    set(subset[subset['x_type'] == 'gene/protein']['x_name'])

    print(f"   (识别到 {len(related_genes)} 个 AD 相关关键基因，正在查找它们之间的连线...)")

    if len(related_genes) > 0:
        # 2. 回到原始大表，查找两端都是这些基因，且关系是 protein_protein 的行
        mask_ppi = (
                (df['x_name'].isin(related_genes)) &
                (df['y_name'].isin(related_genes)) &
                (df['relation'] == 'protein_protein')
        )
        ppi_df = df[mask_ppi]
        print(f"   => 成功补充了 {len(ppi_df)} 条基因互作 (PPI) 边")

        # 3. 合并
        final_df = pd.concat([subset, ppi_df]).drop_duplicates()
    else:
        final_df = subset
        print("   => 未发现相关基因，跳过 PPI 补充。")

    # ================= 保存 =================
    print(f"\n🔹 4. 保存最终 AD 子图至: {OUTPUT_PATH}")
    # 同样不使用 sep='\t'，回归标准 CSV
    final_df.to_csv(OUTPUT_PATH, index=False)

    print("-" * 40)
    print(f"🎉 ADNI 专属知识图谱构建完成！总条目: {len(final_df)}")
    print("   包含: AD疾病节点 + 核心症状(Dementia等) + 关键基因(Amyloid/Tau) + 基因互作网络")
    print("-" * 40)


if __name__ == "__main__":
    main()