import pandas as pd
import numpy as np

# ================= 配置 =================
INPUT_KG = 'kg.csv'             # 你的原始大文件
OUTPUT_KG = 'primekg_ad_subset.csv'  # ★ 生成的 AD 专属子图文件

# ★★★ 关键修正：使用刚才确认的准确名称 ★★★
CORE_NODES = [
    "Alzheimer disease",    # 刚才代码确认找到的 Key
    "Alzheimer's disease"   # 防御性添加，以防 PrimeKG 里混用
]

# =======================================

def extract():
    print(f"1. 正在读取 {INPUT_KG} 并筛选核心节点 {CORE_NODES} 的邻居...")
    chunksize = 100000
    dfs = []

    # --- 第1轮：提取直接相连的边 (1-hop) ---
    # 这一步会把所有和“阿尔茨海默病”有关的基因、药物、通路找出来
    for chunk in pd.read_csv(INPUT_KG, chunksize=chunksize):
        # 筛选 x_name 或 y_name 在核心列表里的边
        mask = chunk['x_name'].isin(CORE_NODES) | chunk['y_name'].isin(CORE_NODES)
        if mask.any():
            dfs.append(chunk[mask])

    if not dfs:
        print("❌ 错误：未提取到任何数据，请再次检查 CSV 列名是否为 x_name/y_name！")
        return

    hop1_df = pd.concat(dfs)
    print(f"   > 1-hop 提取完成，共 {len(hop1_df)} 条关系 (包含药物、基因、通路等)")

    # --- 第2轮：提取扩展关系 (2-hop) ---
    # 这一步是为了让“AD相关基因”和“AD治疗药物”之间如果有相互作用，也能连上
    # 这样图谱才是稠密的，而不是只有中心发散的星形
    print("2. 正在提取 2-hop 内部关联 (这一步能极大丰富语义)...")

    # 获取 1-hop 涉及的所有节点（比如所有相关药物、基因的名字）
    related_nodes = set(hop1_df['x_name']) | set(hop1_df['y_name'])

    dfs_2 = []
    # 再次扫描，提取两端都在“朋友圈”里的边
    for chunk in pd.read_csv(INPUT_KG, chunksize=chunksize):
        mask = chunk['x_name'].isin(related_nodes) & chunk['y_name'].isin(related_nodes)
        if mask.any():
            dfs_2.append(chunk[mask])

    if dfs_2:
        hop2_df = pd.concat(dfs_2)
        final_df = pd.concat([hop1_df, hop2_df]).drop_duplicates()
    else:
        final_df = hop1_df

    print(f"✅ AD 子图提取完成！总计: {len(final_df)} 条三元组。")
    final_df.to_csv(OUTPUT_KG, index=False)
    print(f"💾 已保存至 {OUTPUT_KG}")


if __name__ == "__main__":
    extract()