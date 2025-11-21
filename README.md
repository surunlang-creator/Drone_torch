## 使用指南 / User Guide

### 中文使用指南

#### 1. 环境配置
```bash
# 安装依赖
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn

# 检查PyTorch版本
python -c "import torch; print(torch.__version__)"
```

#### 2. 数据准备
**输入文件格式：**
- **基因表达矩阵**（CSV格式）：行为基因，列为样本
  ```
  Gene_ID,Sample1,Sample2,Sample3,...
  Gene1,5.23,4.67,6.12,...
  Gene2,3.45,3.89,4.01,...
  ```

- **样本分组文件**（TXT/CSV格式）：
  ```
  Sample_ID    Phenotype
  Sample1      worker
  Sample2      drone
  Sample3      queen
  ```

#### 3. 基础运行
```bash
# 最简单的运行方式（使用默认参数）
python ml_pipeline_main.py \
    --expression_matrix gene_expression.csv \
    --sample_labels sample_groups.txt \
    --output_dir ./results

# 带自定义参数的运行
python ml_pipeline_main.py \
    --expression_matrix gene_expression.csv \
    --sample_labels sample_groups.txt \
    --output_dir ./results \
    --test_size 0.2 \
    --val_size 0.2 \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --random_seed 5678
```

#### 4. 高级选项
```bash
# 完整参数示例
python ml_pipeline_main.py \
    --expression_matrix data/expression.csv \
    --sample_labels data/labels.txt \
    --output_dir results/tcn_analysis \
    --model_type tcn \              # 模型类型：tcn/lstm/cnn等
    --test_size 0.2 \               # 测试集比例
    --val_size 0.2 \                # 验证集比例
    --epochs 100 \                  # 最大训练轮数
    --batch_size 32 \               # 批次大小
    --learning_rate 0.001 \         # 初始学习率
    --dropout 0.3 \                 # Dropout比例
    --early_stopping_patience 20 \  # 早停耐心值
    --random_seed 5678 \            # 随机种子
    --use_gpu                       # 使用GPU（如果可用）
```

#### 5. 输出文件说明
程序运行后会在 `output_dir` 生成以下文件：

```
results/
├── models/
│   ├── tcn_best_model.pth          # 最佳模型权重
│   └── tcn_final_model.pth         # 最终模型权重
├── figures/
│   ├── training_curves.png         # 训练/验证损失和准确率曲线
│   ├── confusion_matrix.png        # 混淆矩阵
│   └── feature_importance.png      # 特征重要性排序
├── results/
│   ├── test_predictions.csv        # 测试集预测结果
│   ├── feature_importance.csv      # 基因重要性评分
│   └── training_log.txt            # 训练日志
└── config.json                     # 运行配置记录
```

#### 6. 结果解读
- **training_curves.png**：检查是否过拟合（训练/验证曲线应接近）
- **confusion_matrix.png**：查看各表型分类准确性
- **feature_importance.csv**：获取top基因用于下游验证（qPCR等）

---

### English User Guide

#### 1. Environment Setup
```bash
# Install dependencies
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn

# Verify PyTorch installation
python -c "import torch; print(torch.__version__)"
```

#### 2. Data Preparation
**Input File Formats:**
- **Gene Expression Matrix** (CSV format): Rows are genes, columns are samples
  ```
  Gene_ID,Sample1,Sample2,Sample3,...
  Gene1,5.23,4.67,6.12,...
  Gene2,3.45,3.89,4.01,...
  ```

- **Sample Labels File** (TXT/CSV format):
  ```
  Sample_ID    Phenotype
  Sample1      worker
  Sample2      drone
  Sample3      queen
  ```

#### 3. Basic Usage
```bash
# Simplest run with default parameters
python ml_pipeline_main.py \
    --expression_matrix gene_expression.csv \
    --sample_labels sample_groups.txt \
    --output_dir ./results

# Run with custom parameters
python ml_pipeline_main.py \
    --expression_matrix gene_expression.csv \
    --sample_labels sample_groups.txt \
    --output_dir ./results \
    --test_size 0.2 \
    --val_size 0.2 \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --random_seed 5678
```

#### 4. Advanced Options
```bash
# Full parameter example
python ml_pipeline_main.py \
    --expression_matrix data/expression.csv \
    --sample_labels data/labels.txt \
    --output_dir results/tcn_analysis \
    --model_type tcn \              # Model type: tcn/lstm/cnn, etc.
    --test_size 0.2 \               # Test set ratio
    --val_size 0.2 \                # Validation set ratio
    --epochs 100 \                  # Maximum training epochs
    --batch_size 32 \               # Batch size
    --learning_rate 0.001 \         # Initial learning rate
    --dropout 0.3 \                 # Dropout rate
    --early_stopping_patience 20 \  # Early stopping patience
    --random_seed 5678 \            # Random seed for reproducibility
    --use_gpu                       # Use GPU if available
```

#### 5. Output Files
After execution, the following files will be generated in `output_dir`:

```
results/
├── models/
│   ├── tcn_best_model.pth          # Best model checkpoint
│   └── tcn_final_model.pth         # Final model checkpoint
├── figures/
│   ├── training_curves.png         # Training/validation loss and accuracy curves
│   ├── confusion_matrix.png        # Confusion matrix
│   └── feature_importance.png      # Feature importance ranking
├── results/
│   ├── test_predictions.csv        # Test set predictions
│   ├── feature_importance.csv      # Gene importance scores
│   └── training_log.txt            # Training log
└── config.json                     # Configuration record
```

#### 6. Result Interpretation
- **training_curves.png**: Check for overfitting (train/val curves should be close)
- **confusion_matrix.png**: Examine classification accuracy per phenotype
- **feature_importance.csv**: Extract top genes for downstream validation (qPCR, etc.)

---
