# ZeroShot_ImmuneFeature_Drift: 基于零样本 AI 基础模型的纵向 iPOP 监测：通过 UMI 免疫衰老指纹图谱实现

### 1. 背景

**Immune-Drift-Zero** 展示了如何以严谨、可重复的方式，将现有 AI 基础模型（Transformer 编码器：scGPT-blood）适配到新的生物学问题：从纵向 Bulk PBMC 数据中追踪个体内免疫漂移。

#### 1.1 科学创新点

- **创新性 stLFR 共条码技术伪影的 ML QC ** 现有 stLFR/UMI Pipeline 普遍缺乏对共条码分布异常的数据驱动 QC（PCR 偏好、光学重复等）。本项目实现了跨时间点联合训练的 **Isolation Forest** 模型用于异常聚类检测（测试结果：各时间点异常率 8.5–11.9%，异常聚类平均计数为中位数的 10–100 倍）。这填补了 stLFR 领域文献中的一个已知空白。

- **低成本 Isoform 代理。** 以 MGI stLFR 共条码短读长替代昂贵的 PacBio/ONT 长读长测序，通过共条码聚类实现伪全长 Isoform 分辨率，成本等同于标准 Illumina 测序。

- **零样本基础模型应用。** 将预训练 `scGPT-blood`（1030 万单细胞）应用于纵向 Bulk RNA-seq 数据——将 scGPT 超越其单细胞设计边界（参考 Kedzierska et al., 2024）。在 N=3, T=1 的极小样本条件下，零样本方法是唯一可行的无过拟合方案。

- **加法剪接特征融合。** 共条码香农熵和主导聚类比例经线性投影后，在 Transformer 编码器层之前与基因 Embedding **加法融合**——无需修改词表或微调权重即可保留剪接多样性信息。

#### 1.2 为什么关注免疫漂移 → 衰老推断

衰老是生物医学研究的核心挑战之一，驱动癌症、神经退行性疾病、心血管疾病和免疫衰退的风险。免疫系统是**生物年龄的实时生物标志物**：免疫细胞组成的纵向变化、T 细胞耗竭标记物和剪接异构体转换（如 PTPRC/CD45 异构体比率）直接反映免疫衰老（immunosenescence）——随年龄增长免疫功能的进行性退化（López-Otín et al., 2013; Goronzy & Weyand, 2019）。

本项目基于我在衰老机制研究方面的经验。在博士期间, 我参与了人类衰老基因的系统水平分析，发现衰老相关基因在生物网络中倾向于占据枢纽位置，具有独特的网络结构特征和组织特异性表达模式（Zhang, Nogales-Cadenas, ..., **Cai Y**, Vijg J, Zhang ZD. *Human Molecular Genetics*, 2016, 25(14):2934–2947）。该研究揭示衰老基因并非随机分布，而是占据生物网络中结构关键位置——这启发了当前使用基础模型嵌入（从 1030 万细胞中编码基因-基因交互模式）来检测免疫基因网络微妙时序漂移的方法。

通过零样本 AI 监测**个体内免疫轨迹**随时间的变化，本 Pipeline 将分子衰老研究与临床纵向监测（iPOP）连接起来，实现免疫衰老的早期检测，无需在每个时间点进行昂贵的单细胞或长读长测序。

#### 1.3 关键量化结果

**Iteration 8 — 真实 scGPT-blood 模型 + PBMC 数据：**

| 指标 | 数值 |
|---|---|
| 测试时间点 | 3 个（2024–2026；2024 = 真实 chr1-PTPRC，2025/2026 = 模拟漂移）|
| 每样本基因数 | 1420–1509（QC 过滤后）|
| 匹配 scGPT 词表的基因数 | 559–606 / 1462 |
| QC 异常率 | 9.8–10.1% |
| **真实 scGPT** 余弦相似度 | 0.999997–0.999999 |
| **真实 scGPT** 欧氏漂移 | 每年 0.039–0.059 |
| **MockScGPT** 余弦相似度 | 1.000000–0.999997 |
| **MockScGPT** 欧氏漂移 | 每年 0.013–0.042 |

> **解读：** 真实 scGPT-blood 嵌入对时序漂移的灵敏度更高（欧氏距离 0.039–0.059 vs MockScGPT 0.013–0.042），同时保持 >0.9999 的余弦相似度反映个体免疫稳定性。12 层 Transformer 从 1030 万预训练细胞中学到的基因-基因交互模式比随机权重更能捕捉细微变化。

---

### 2. 材料与方法

#### 2.1 输入与输出

**输入：**
- 原始数据：MGI stLFR FASTQ（共条码短读长），或 CSV 清单文件（列：`sample_id, fastq_r1, fastq_r2`）。
- 预处理输入：基因级共条码计数表 `(gene_id, co_barcode_cluster_id, read_count)`。
- 标准 Bulk RNA-seq：featureCounts 输出可通过 `stlfr_preprocess.py` 转换为共条码格式。

**输出：**
- `qc/qc_*.json`：逐样本 QC（异常聚类、异常分数、泊松随机性检验）。
- `qc/qc_summary.csv`：跨样本 QC 汇总。
- `figures/drift_metrics.csv`：每次年度转换的余弦相似度与欧氏距离。
- `figures/embedding_trajectory.png`：跨时间点 scGPT 零样本 Embedding 的 PCA 轨迹。
- `figures/cell_proportions.png`：免疫细胞亚群漂移堆叠柱状图。
- `figures/splicing_fingerprint.png`：核心免疫标记基因 Isoform Switch 热图。

#### 2.2 Pipeline 步骤（详见英文版）

管道包含 5 个顺序步骤：步骤 -1（stLFR 预处理）→ 步骤 0（ML QC）→ 步骤 1（特征工程）→ 步骤 2（零样本 scGPT Embedding）→ 步骤 3（CIBERSORTx 去卷积）+ 步骤 4（漂移分析与可视化）。

#### 2.3 方法论选择理由

| 选择 | 替代方案 | 理由 |
|---|---|---|
| stLFR 共条码 | 标准 Bulk RNA-seq | 保留长距离外显子连接信息，实现伪 Isoform 分辨率 |
| 零样本 scGPT | 微调 / 定制 Transformer | N=3, T=1 必然导致过拟合；零样本利用 1030 万细胞的免疫流形 |
| Isolation Forest QC | 固定百分位阈值 | 数据驱动异常检测，跨时间点阈值一致，无需硬编码 |
| 加法特征融合 | 扩展词表 | 保留预训练权重，无需 GPU/微调 |

---

### 3. 讨论与待办事项

#### 3.1 零样本 Bulk 投射：已知局限性与缓解策略

Kedzierska et al. (2024, arXiv:2403.11375) 证明**直接将 bulk RNA-seq 零样本投射到 scGPT 潜空间效果有限**，因为 scGPT 在单细胞数据上预训练，latent space 的"插值区域"对 bulk 数据（数千细胞的平均值）泛化能力差。他们提出 MLP-A smoothing module（用 mixup 正则化训练的 3 层 MLP）来平滑潜空间。

**本项目的 additive diversity fusion 解决的是不同层面的问题**：在编码器之前丰富基因 embedding 的剪接信息，但未解决编码器之后的 SC→Bulk 分布偏移。

**为什么漂移检测仍然有效**：对于本项目的核心任务——追踪同一个体 2024→2026 的时序漂移——bulk 投射的系统性偏差在所有时间点上是**一致**的。余弦相似度和欧氏距离度量的是**相对变化**，不是绝对语义位置。只要所有样本的系统偏移相同，相对距离就是有效的漂移指标。

#### 3.2 CIBERSORTx：为什么使用模拟结果

| 工具 | 方法 | 优点 | 缺点 |
|---|---|---|---|
| **CIBERSORTx** | 线性 SVM | 金标准，LM22 签名矩阵权威，大量文献引用 | 需要 Docker + 注册 token |
| MuSiC | 加权非负最小二乘 | 适合单细胞参考 | 对 bulk 数据质量敏感 |
| xCell | 线性混合模型 | 在线工具，无需安装 | 精度较低 |
| EPIC | 约束最小二乘 | 肿瘤数据友好 | 免疫细胞类型较少 |
| Scaden | 深度学习 | 端到端训练 | 需要大量训练数据 |

**当前状态**：CIBERSORTx 结果使用随机 Dirichlet 模拟。原因是 CIBERSORTx 需要在 [cibersortx.stanford.edu](https://cibersortx.stanford.edu/) 注册获取 API token，作为商业公司研究人员，token 申请未获批准。Pipeline 已生成标准 CIBERSORTx 输入格式（CPM 归一化的 genes × samples mixture file），获得 token 后可直接提交运行。

#### 3.3 隐私 / HIPAA 合规

本 demo 使用人类 PBMC 数据（ch1-PTPRC 子集）。所有展示的输出均为**聚合统计摘要**（PCA embedding、漂移指标、细胞比例），不包含原始序列数据。输出文件中不含 HIPAA 定义的 18 项受保护健康信息（PHI）标识符。基因计数表仅包含基因名和整数计数，无患者标识。


**待办事项（优先级排序）：**
- [ ] **真实 CIBERSORTx 运行：** Mixture file 已就绪；需要 Docker + [cibersortx.stanford.edu](https://cibersortx.stanford.edu/) 注册 token。
- [ ] **真实 stLFR 数据：** 将 in-house stLFR PBMC 共条码计数矩阵（含真实共条码，非模拟单聚类/基因）接入 pipeline。
- [ ] **批次效应校正：** 集成 ComBat/Harmony。
- [ ] **纵向统计检验：** 配对检验或混合效应模型。
