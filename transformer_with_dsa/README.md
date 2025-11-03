# Transformer with DSA (Dynamic Sparse Attention)

这是一个集成了DSA算法优化的Transformer模型训练系统。

## 特性

- **Dynamic Sparse Attention (DSA)**: 动态稀疏注意力算法，优化Transformer的训练效率
- **断点续训**: 支持从上次训练中断的地方继续训练
- **自动模型保存**: 每隔指定epoch自动保存模型检查点
- **灵活配置**: 通过修改启动脚本内的变量即可配置所有参数
- **完整日志**: 详细的训练日志和可视化

## 文件结构

```
transformer_with_dsa/
├── python/                    # 核心Python代码
│   ├── dsa_algorithm.py      # DSA算法实现
│   ├── transformer_dsa.py    # 集成DSA的Transformer模型
│   ├── data_loader.py        # 数据加载器
│   └── trainer_dsa.py        # 训练器（含断点续训）
├── example/                   # 启动脚本和配置
│   ├── run_training.sh       # 主训练脚本
│   ├── quick_test.sh         # 快速测试脚本
│   ├── log/                  # 日志保存目录
│   └── model/                # 模型保存目录
└── README.md                 # 说明文档
```

## 快速开始

### 1. 快速测试

运行快速测试来验证系统：

```bash
cd transformer_with_dsa/example
./quick_test.sh
```

这将运行一个5个epoch的快速训练，验证系统是否正常工作。

### 2. 完整训练

修改 `run_training.sh` 中的配置参数，然后运行：

```bash
cd transformer_with_dsa/example
./run_training.sh
```

### 3. 断点续训

如果训练意外中断，设置 `RESUME_TRAINING=true` 然后重新运行：

```bash
# 在 run_training.sh 中修改
RESUME_TRAINING=true

# 然后运行
./run_training.sh
```

## 配置参数说明

### 基本训练参数

- `NUM_EPOCHS`: 训练轮数
- `BATCH_SIZE`: 批大小
- `LEARNING_RATE`: 学习率
- `SAVE_EVERY`: 每隔多少epoch保存模型（默认50）
- `RESUME_TRAINING`: 是否从上次训练继续

### 模型参数

- `INPUT_DIM`: 输入特征维度
- `EMBED_DIM`: 嵌入维度
- `NUM_HEADS`: 注意力头数
- `NUM_LAYERS`: Transformer层数
- `FF_DIM`: 前馈网络隐藏层维度
- `OUTPUT_DIM`: 输出维度

### DSA参数

- `DSA_ENABLED`: 是否启用DSA优化
- `SPARSITY_RATIO`: 初始稀疏度比例（如0.1表示保留10%的连接）
- `TARGET_SPARSITY`: 目标稀疏度
- `SPARSITY_WEIGHT`: 稀疏度正则化权重

### 数据参数

- `DATASET_PATH`: 数据集路径
- `SEQ_LEN`: 序列长度（留空自动确定）
- `TRAIN_RATIO`: 训练集比例
- `NORMALIZE`: 是否归一化数据

## DSA算法说明

DSA (Dynamic Sparse Attention) 是一种优化Transformer注意力机制的方法：

1. **动态稀疏化**: 根据重要性动态选择稀疏的attention连接
2. **自适应调度**: 训练过程中自动调整稀疏度
3. **信息保留**: 在稀疏化的同时保留关键信息
4. **梯度友好**: 稀疏化过程对梯度训练友好

## 输出文件

### 模型文件

保存在 `example/model/[timestamp]/` 目录下：

- `checkpoint_epoch_X.pth`: 每隔指定epoch的检查点
- `final_model.pth`: 最终训练完成的模型
- `checkpoint_info.json`: 检查点信息文件
- `training_curves.png`: 训练曲线图

### 日志文件

保存在 `example/log/` 目录下：

- `training_YYYYMMDD_HHMMSS.log`: 详细的训练日志

## 数据格式

支持多种数据格式：

- `.npy`: NumPy数组文件
- `.pkl`: Pickle文件
- `.csv`: CSV文件
- `.json`: JSON文件

数据应该是一个二维数组 `[N, D]`，其中 N 是样本数，D 是特征维度。

## 示例配置

### 回归任务（默认）

```bash
TASK_TYPE="regression"
OUTPUT_DIM=1
```

### 分类任务

```bash
TASK_TYPE="classification"
OUTPUT_DIM=10  # 10个类别
```

### 大模型配置

```bash
EMBED_DIM=512
NUM_HEADS=16
NUM_LAYERS=8
FF_DIM=2048
BATCH_SIZE=64
```

### 高稀疏度配置

```bash
DSA_ENABLED=true
SPARSITY_RATIO=0.05    # 保留5%连接
TARGET_SPARSITY=0.01   # 目标1%连接
```

## 常见问题

### Q: 如何调整DSA的稀疏度？

A: 修改 `SPARSITY_RATIO`（初始稀疏度）和 `TARGET_SPARSITY`（目标稀疏度）参数。数值越小，稀疏度越高。

### Q: 训练中断了怎么办？

A: 设置 `RESUME_TRAINING=true`，系统会自动从最新的检查点继续训练。

### Q: 如何禁用DSA？

A: 设置 `DSA_ENABLED=false`，系统将使用标准的Transformer注意力机制。

### Q: 如何调整保存频率？

A: 修改 `SAVE_EVERY` 参数，例如设置为10表示每10个epoch保存一次。

## 性能优化建议

1. **GPU使用**: 确保安装了CUDA版本的PyTorch
2. **批大小**: 根据GPU内存调整 `BATCH_SIZE`
3. **数据加载**: 调整 `NUM_WORKERS` 优化数据加载速度
4. **稀疏度**: 从较低的稀疏度开始（如0.2），逐步增加

## 系统要求

- Python 3.7+
- PyTorch 1.8+
- NumPy
- scikit-learn
- matplotlib
- pandas

## 安装依赖

```bash
pip install torch numpy scikit-learn matplotlib pandas
```