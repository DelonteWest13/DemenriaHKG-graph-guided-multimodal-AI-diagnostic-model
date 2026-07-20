# Knowledge-Augmented Multimodal Synergistic Interaction

这是论文“Knowledge-Augmented Multimodal Synergistic Interaction”对应的源码仓库，重点面向神经退行性疾病诊断任务，融合了影像数据、患者临床记录与知识图谱信息，覆盖了四个常用队列：ADNI、NACC、PPMI 和 AIBL。

## 1. 项目简介

本项目的核心目标是：

- 结合多模态信息进行疾病分类与诊断建模；
- 将医学知识图谱引入多模态学习过程，增强模型的可解释性与泛化能力；
- 对比不同知识增强策略，包括：
  - 基于私域知识图谱 CustomKG 的方法；
  - 基于精炼疾病相关子图 DementiaHKG（论文中也可称为 DemRKG）的方法；
  - 基于医学知识库 PrimeKG 的方法；
  - 以及基于双模态、三模态与知识引导 Transformer/Perceiver 的模型结构。

## 2. 仓库结构概览

仓库根目录下主要包含四个数据集目录：

- [ADNI](ADNI/)
- [AIBL](AIBL/)
- [NACC](NACC/)
- [PPMI](PPMI/)

每个数据集目录都包含以下几类内容：

1. 患者记录与标签文件（数据集名称.csv）
   - 例如 ADNI 中的 ADNI.csv、ADNI诊断.csv、AD.csv、mci.csv、normal.csv；
   - AIBL 中的 AIBL.csv、AD.csv、MCI.csv、NC.csv；
   - NACC 中的 NACC_ad.csv、NACC_mci.csv、NACC_normal.csv；
   - PPMI 中的 control.csv、PD1.csv、prodromal.csv、swedd.csv 等。

2. 影像数据
   - 主要为 NIfTI 格式影像文件；
   - 数据集目录下有 MRI 或相关影像子目录；
   - 有影像的压缩包备份

3. 下游分类实验脚本与 Notebook
   - 每个数据集目录下都包含多种实验 Notebook，例如：
     - 双模态实验（EHR+MRI，MM-Transformer）；
     - 三模态实验（EHR+MRI+BM，MM-Transformer）；
     - 常规三模态+K引导（EHR+MRI+BM+KG（TransE），知识拼接型MM-Transformer）
     - 基于 CustomKG 的实验；
     - 基于 DementiaHKG 的实验；
     - 基于 PrimeKG 的实验；
     - 以及包含 KAMP/KAMT 结构的实验。

4. 模型定义文件
   - 每个数据集目录下都包含 [model1.py](ADNI/model1.py) 这类模型定义脚本；
   - 其中定义了图像编码器、临床特征编码器、Transformer/Perceiver 风格的多模态融合模块，以及知识图谱增强模块。

## 3. 数据集说明

### 3.1 ADNI

ADNI 数据集目录包含：

- 结构化患者记录文件，例如 [ADNI/ADNI.csv](ADNI/ADNI.csv) 和 [ADNI/ADNI诊断.csv](ADNI/ADNI诊断.csv)
- 影像数据目录 [ADNI/MRI](ADNI/MRI)
- 下游分类实验 Notebook，如 [ADNI/ADNI三模态+CustomKG.ipynb](ADNI/ADNI三模态+CustomKG.ipynb)、[ADNI/ADNI三模态+DementiaHKG.ipynb](ADNI/ADNI三模态+DementiaHKG.ipynb)、[ADNI/ADNI三模态+PrimeKG.ipynb](ADNI/ADNI三模态+PrimeKG.ipynb)
- 模型定义文件 [ADNI/model1.py](ADNI/model1.py)
- 知识图谱相关目录 [ADNI/CustomKG](ADNI/CustomKG)、[ADNI/DementiaHKG](ADNI/DementiaHKG)、[ADNI/PrimeKG](ADNI/PrimeKG)
- 训练 CustomKG 的脚本 [ADNI/ADNI-train-CustomKG.py](ADNI/ADNI-train-CustomKG.py)

### 3.2 AIBL

AIBL 数据集目录包含：

- 患者记录文件，如 [AIBL/AIBL.csv](AIBL/AIBL.csv)、[AIBL/AD.csv](AIBL/AD.csv)、[AIBL/MCI.csv](AIBL/MCI.csv)、[AIBL/NC.csv](AIBL/NC.csv)
- 影像数据子目录 [AIBL/AD](AIBL/AD)、[AIBL/MCI](AIBL/MCI)、[AIBL/NC](AIBL/NC)
- 实验 Notebook，如 [AIBL/AIBL三模态+CustomKG.ipynb](AIBL/AIBL三模态+CustomKG.ipynb)、[AIBL/AIBL三模态+DementiaHKG.ipynb](AIBL/AIBL三模态+DementiaHKG.ipynb)、[AIBL/AIBL三模态+PrimeKG.ipynb](AIBL/AIBL三模态+PrimeKG.ipynb)
- 模型定义文件 [AIBL/model1.py](AIBL/model1.py)
- 知识图谱目录 [AIBL/CustomKG](AIBL/CustomKG)、[AIBL/DementiaHKG](AIBL/DementiaHKG)、[AIBL/PrimeKG](AIBL/PrimeKG)

### 3.3 NACC

NACC 数据集目录包含：

- 患者记录文件，如 [NACC/NACC_ad.csv](NACC/NACC_ad.csv)、[NACC/NACC_mci.csv](NACC/NACC_mci.csv)、[NACC/NACC_normal.csv](NACC/NACC_normal.csv)
- 影像数据目录 [NACC/NACC_nii_ad](NACC/NACC_nii_ad)、[NACC/NACC_mci](NACC/NACC_mci)
- 实验 Notebook，如 [NACC/NACC三模态.ipynb](NACC/NACC三模态.ipynb)、[NACC/NACC三模态+CustomKG.ipynb](NACC/NACC三模态+CustomKG.ipynb)、[NACC/NACC三模态+DementiaHKG.ipynb](NACC/NACC三模态+DementiaHKG.ipynb)
- 模型定义文件 [NACC/model1.py](NACC/model1.py)
- 知识图谱目录 [NACC/CustomKG](NACC/CustomKG)、[NACC/DementiaHKG](NACC/DementiaHKG)、[NACC/PrimeKG](NACC/PrimeKG)

### 3.4 PPMI

PPMI 数据集目录包含：

- 患者记录文件，如 [PPMI/PPMI诊断.csv](PPMI/PPMI诊断.csv)、[PPMI/control.csv](PPMI/control.csv)、[PPMI/PD1.csv](PPMI/PD1.csv)、[PPMI/prodromal.csv](PPMI/prodromal.csv)
- 影像数据目录 [PPMI/Control](PPMI/Control)、[PPMI/PD](PPMI/PD)、[PPMI/Prodromal](PPMI/Prodromal)、[PPMI/SWEDD](PPMI/SWEDD)
- 实验 Notebook，如 [PPMI/PPMI三模态.ipynb](PPMI/PPMI三模态.ipynb)、[PPMI/PPMI三模态+CustomKG.ipynb](PPMI/PPMI三模态+CustomKG.ipynb)、[PPMI/PPMI三模态+DementiaHKG.ipynb](PPMI/PPMI三模态+DementiaHKG.ipynb)、[PPMI/PPMI三模态+PrimeKG.ipynb](PPMI/PPMI三模态+PrimeKG.ipynb)
- 模型定义文件 [PPMI/model1.py](PPMI/model1.py)
- 知识图谱目录 [PPMI/CustomKG](PPMI/CustomKG)、[PPMI/DementiaHKG](PPMI/DementiaHKG)、[PPMI/PrimeKG](PPMI/PrimeKG)

## 4. 知识图谱模块说明

本项目的知识增强部分主要由三类知识来源构成：

### 4.1 CustomKG

CustomKG 是基于数据集或领域知识构建的私域知识图谱，主要用于：

- 组织患者、疾病、临床特征与医学实体之间的关系；
- 为多模态模型提供结构化先验知识；
- 支持知识引导下的特征融合与分类。

### 4.2 DementiaHKG / DemRKG

DementiaHKG 是从医学知识库中精炼得到的与痴呆相关的子图，通常用于：

- 聚焦与痴呆、认知障碍、神经退行性疾病相关的关键实体和关系；
- 作为疾病相关先验知识增强模型；
- 在论文中也可被视为 DemRKG 这一更贴近论文表述的名称。

### 4.3 PrimeKG

PrimeKG 是医学知识图谱的基础来源之一，仓库中对应目录下会包含实体/关系映射文件、知识图谱相关中间产物，以及知识图谱原始文件（如 kg.csv）。

## 5. 主要代码组件

### 5.1 模型定义

各数据集目录下的 [model1.py](ADNI/model1.py) 主要包含：

- NIfTI 影像读取与裁剪逻辑；
- 3D CNN 图像编码器；
- 多模态融合模块；
- Transformer / Perceiver 风格的知识引导融合层；
- 与知识图谱嵌入相连的分类头。

### 5.2 实验 Notebook

Notebook 主要用于：

- 数据预处理；
- 构造样本对；
- 加载知识图谱嵌入；
- 训练/验证/测试分类模型；
- 输出结果与可视化分析。

### 5.3 知识图谱训练脚本

在 ADNI 中提供了 [ADNI/ADNI-train-CustomKG.py](ADNI/ADNI-train-CustomKG.py)，用于生成和训练与知识图谱相关的中间文件。其他数据集也可基于相同思路扩展。

## 6. 运行建议

由于该项目主要以 Notebook 为主，推荐按以下方式使用：

1. 选择目标数据集目录，如 [ADNI](ADNI/) 或 [PPMI](PPMI/)
2. 确保对应的患者表格 CSV 文件和影像数据已经准备好
3. 进入对应目录后运行相应的实验 Notebook
4. 若需要重建知识图谱或嵌入，使用对应目录下的知识图谱脚本或 Notebook

> 说明：由于本仓库包含较多实验版本和中间产物，实际运行时可能需要根据本地数据路径进行少量调整。数据集链接：

## 7. 依赖环境

建议使用 Python 3.8 及以上版本。常见依赖包括：

- PyTorch
- nibabel
- pandas
- numpy
- scikit-learn
- tqdm
- 以及在知识图谱训练阶段可能需要的图学习/知识图谱相关依赖

## 8. 说明与后续扩展

- 本仓库偏向论文复现与实验研究代码，结构较为灵活，部分文件为实验版本或中间产物；
- 如果后续需要进一步整理为正式可复现流水线，可以继续补充：
  - 统一的数据预处理脚本；
  - 统一的训练入口；
  - 配置文件；
  - 结果日志与实验表格导出脚本。

