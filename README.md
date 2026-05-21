## XJTU-SY 滚动轴承端到端在线诊断流水线

本目录实现基于 **XJTU-SY Bearing Datasets** 的端到端 PHM 流水线：

> 特征提取 → 故障诊断 → 健康指标构造 → 剩余寿命预测

所有代码均位于 `src/`，输出写入项目根目录 `output/`。

---

### 1. 模块结构

```text
src/
  __init__.py
  config.py              # 全局配置、轴承元数据、故障频率公式
  data_loader.py         # CSV 读取 + 严格按数字顺序的窗口生成器
  feature_engineering.py # 时域/频域/小波/差分特征
  fault_diagnosis.py     # TabPFN 分类、物理弱标签、两轮推理、SHAP、在线评估
  health_index.py        # SHAP/等权重 HI 构造、健康分级、预警时效评分
  rul.py                 # 单调化 HI、零样本/小样本 RUL 预测、置信区间
  run.py                 # CLI 入口
```

---

### 2. 设计说明

#### 2.1 配置中心：`config.py`

- 硬编码 15 条轴承工况元数据
- 实现 `compute_fault_freqs(speed_hz)`
- 集中管理：
  - 采样率、窗口比例 `p`
  - HI 参数、RUL 参数
  - TabPFN 本地 checkpoint 路径
  - SHAP 参数、物理规则阈值、小样本上下文参数

#### 2.2 数据加载：`data_loader.py`

- 按 `1.csv -> 2.csv -> ...` 的**数字顺序**读取
- 每个窗口返回 `WindowRecord`
- 不跨轴承混合，仅在单一轴承目录内按时间推进
- `count_windows()` 用于在线模块估计总寿命长度，服务于健康参考窗口比例和预警时效评分

#### 2.3 特征工程：`feature_engineering.py`

对水平/竖直通道提取：

- 时域：RMS、Peak、MAV、Kurtosis、Crest、Impulse、Skew
- 频域：
  - Hann 窗 + RFFT
  - BPFO/BPFI/BSF/FTF 邻域幅值
  - 频谱重心、谱熵
- 小波：
  - 默认 `db4`、4 层
  - 各层能量比与小波熵
- 跨通道：`rms_combined`、`kurt_max`、`rho_hv`、`energyratio_hv`
- 差分：所有基础特征的一阶差分

#### 2.4 故障诊断：`fault_diagnosis.py`

支持两种模式：

1. **0 样本模式**
   - 前 10% 窗口作为健康参考
   - 基于 RMS 损伤阈值触发故障检测
   - 基于 BPFI/BPFO/FTF/BSF 邻域频谱幅值构造物理弱标签
   - 若多个故障频率同时显著，则标记为混合故障并记录组合

2. **小样本模式**
   - 使用 `--support-bearings` 指定参考轴承
   - 读取 support bearings 全生命周期特征
   - 仅取后 30% 退化阶段按步长采样构造 support context
   - 标签直接来自元数据 Ground Truth

诊断推理采用**两轮 TabPFN**：

- 第一轮：原始特征 + 物理/真实上下文
- 第二轮：加入第一轮概率与滞后概率特征再次推理

同时实现：

- 在线混淆矩阵
- Accuracy / Macro-F1 / Weighted-F1 / Brier Score
- 5 段校准曲线数据
- `KernelExplainer` 的局部解释；若不可用则自动退化为因果 z-score 解释

> 注意：SHAP 仅做解释与 HI 加权参考，**不修改** `predict_proba` 的最终输出。

#### 2.5 健康指标：`health_index.py`

- 按特征方向构造单特征损伤度
- 优先采用 SHAP 绝对值归一化权重；若不可用则等权
- `HI = exp(-lambda * D)`
- 输出：`HI`、`D`、健康等级、权重、各特征损伤度
- 附加实现预警时效分数 `Score_timing`

#### 2.6 RUL 预测：`rul.py`

- 先对 HI 做 EMA + 运行最小值，得到因果单调 HI
- **0 样本模式**：
  - 线性跨阈值
  - 线性外推
  - 指数衰减兜底
- **小样本模式**：
  - 将最近 `L` 个单调 HI 组成表格特征窗口
  - 以 support bearings 生成 `(HI-window, RUL)` 支持集
  - 用 `TabPFNRegressor` 输出当前 RUL
- 输出：点预测、CI、方法名、未来 HI 轨迹

---

### 3. 依赖建议

```bash
pip install numpy shap pywavelets torch tabpfn
```

若 `tabpfn` / `shap` / `pywt` 不可用，代码会尽量自动降级，但完整功能建议安装齐全。

---

### 4. 用法

```bash
# 零样本在线诊断
python -m src.run --bearings Bearing1_1 --device auto

# 快速调试：仅跑前 5 个窗口
python -m src.run --bearings Bearing1_1 --max-windows 5 --device cpu

# 小样本模式：指定参考轴承
python -m src.run --bearings Bearing3_3 --support-bearings Bearing1_1 Bearing2_1 --device auto

# 调整窗口比例和小波基
python -m src.run --bearings Bearing3_2 --p 0.5 --wavelet sym4
```

---

### 5. 输出文件

流水线会在 `output/` 中生成：

- `features/*_features.csv`：历史特征表
- `diagnosis/*_diagnosis.csv`：类别概率、弱标签、SHAP、在线指标
- `hi/*_hi.csv`：HI、D、健康等级、预警时效评分
- `rul/*_rul.csv`：RUL、置信区间、预测方法、未来 HI 轨迹
- `logs/*.log`：运行日志

---

### 6. 关键约束落实情况

- **因果性**：所有状态估计仅依赖当前与历史窗口
- **顺序性**：严格按数字文件名排序
- **不跨轴承混合**：主推理只使用当前轴承；few-shot 仅通过显式 support set 引入参考上下文
- **TabPFN 规范**：分类概率必须由 `predict_proba` 给出；SHAP 只解释不干预
- **设备切换**：支持 `--device auto/cuda/cpu`
