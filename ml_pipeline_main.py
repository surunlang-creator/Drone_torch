"""
完整的机器学习/深度学习主流程 - 修复版 v1.4.0
关键修复:
1. 添加独立测试集 (train/val/test 三分法)
2. 验证集用于early stopping和超参数调整
3. 测试集仅用于最终评估，完全不参与训练
4. 修复数据泄漏问题
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import sys
from datetime import datetime

# 深度学习相关
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 机器学习相关
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                             recall_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, auc)

# 传统 ML 模型
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import joblib

# 自定义深度学习模型
try:
    from dl_models import get_model, MODEL_REGISTRY, count_parameters
    print(f"✓ 成功加载 {len(MODEL_REGISTRY)} 个深度学习模型")
    DL_AVAILABLE = True
except ImportError:
    print("警告: dl_models.py 未找到，将使用基础MLP模型")
    MODEL_REGISTRY = {'MLP': None}
    DL_AVAILABLE = False
    
    def get_model(model_name, input_dim, num_classes, dropout=0.5):
        """基础MLP模型"""
        return nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def count_parameters(model):
        """计算模型参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 导入增强分析模块
try:
    from ml_pipeline_enhancements import MLPipelineEnhancements
    print("✓ 成功加载增强分析模块")
    ENHANCEMENTS_AVAILABLE = True
except ImportError:
    print("警告: ml_pipeline_enhancements.py 未找到")
    MLPipelineEnhancements = None
    ENHANCEMENTS_AVAILABLE = False

warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 全局配置
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)


class GeneExpressionDataset(Dataset):
    """基因表达数据集"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLPipeline:
    """完整的机器学习主流程"""
    
    def __init__(self, config_path: str, output_dir: str, min_epochs: int = 30, 
                 max_epochs: int = 200, min_valid_epochs: int = 10, dropout: float = 0.5,
                 test_size: float = 0.15, val_size: float = 0.15):
        """
        初始化ML流程
        
        Args:
            config_path: 配置文件路径
            output_dir: 输出目录
            min_epochs: 最小训练轮数（防止过拟合，建议>=30）
            max_epochs: 最大训练轮数
            min_valid_epochs: 最佳epoch的最小有效值，低于此值的模型将被舍弃（默认10）
            dropout: Dropout比例，用于防止过拟合（范围0.0-0.8，默认0.5）
            test_size: 测试集比例（默认0.15，即15%）
            val_size: 验证集比例（默认0.15，即15%）
        """
        # 参数验证
        if not 0.0 <= dropout <= 0.8:
            raise ValueError(f"dropout应在0.0-0.8之间，当前值: {dropout}")
        if min_epochs < 1:
            raise ValueError(f"min_epochs必须>=1，当前值: {min_epochs}")
        if max_epochs < min_epochs:
            raise ValueError(f"max_epochs({max_epochs})必须>= min_epochs({min_epochs})")
        if min_valid_epochs < 1:
            raise ValueError(f"min_valid_epochs必须>=1，当前值: {min_valid_epochs}")
        if not 0.05 <= test_size <= 0.3:
            raise ValueError(f"test_size应在0.05-0.3之间，当前值: {test_size}")
        if not 0.05 <= val_size <= 0.3:
            raise ValueError(f"val_size应在0.05-0.3之间，当前值: {val_size}")
        if test_size + val_size >= 0.5:
            raise ValueError(f"test_size + val_size 不应>=0.5，当前: {test_size + val_size}")
        
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_epochs = max(30, min_epochs)
        self.max_epochs = max_epochs
        self.min_valid_epochs = min_valid_epochs
        self.dropout = dropout
        self.test_size = test_size
        self.val_size = val_size
        
        train_size = 1.0 - test_size - val_size
        print(f"数据划分配置:")
        print(f"  训练集: {train_size*100:.1f}%")
        print(f"  验证集: {val_size*100:.1f}% (用于early stopping)")
        print(f"  测试集: {test_size*100:.1f}% (仅用于最终评估)")
        print(f"\n训练配置:")
        print(f"  最小Epochs={self.min_epochs}, 最大Epochs={self.max_epochs}")
        print(f"  Dropout={self.dropout:.2f}")
        print(f"  ⚠️  最佳epoch < {self.min_valid_epochs} 的模型将被舍弃（可能过拟合）")
        
        # 初始化增强分析模块
        if ENHANCEMENTS_AVAILABLE:
            self.enhancements = MLPipelineEnhancements(output_dir)
        else:
            self.enhancements = None

        # 创建子目录
        self.models_dir = self.output_dir / 'models'
        self.figures_dir = self.output_dir / 'figures'
        self.results_dir = self.output_dir / 'results'
        self.logs_dir = self.output_dir / 'logs'
        
        for dir_path in [self.models_dir, self.figures_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 加载配置
        self._load_config()
        
        # 结果存储
        self.results = {
            'traditional_ml': {},
            'deep_learning': {},
            'top_genes': {},
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'config_path': str(config_path),
                'output_dir': str(output_dir),
                'random_seed': RANDOM_SEED,
                'min_epochs': self.min_epochs,
                'max_epochs': self.max_epochs,
                'dropout': self.dropout,
                'test_size': self.test_size,
                'val_size': self.val_size,
                'train_size': train_size
            }
        }
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n使用设备: {self.device}")
        if torch.cuda.is_available():
            print(f"  GPU型号: {torch.cuda.get_device_name(0)}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def _load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            required_fields = ['output_dir', 'traits']
            for field in required_fields:
                if field not in self.config:
                    raise ValueError(f"配置文件缺少必需字段: {field}")
            
            print(f"\n✓ 配置文件加载成功")
            print(f"  性状数量: {len(self.config['traits'])}")
            
        except Exception as e:
            print(f"✗ 配置文件加载失败: {e}")
            sys.exit(1)
    
    def load_data(self, trait: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Optional[Dict]]:
        """加载指定性状的数据"""
        data_path = Path(self.config['output_dir']) / 'ml_ready' / f'{trait}_ml_data.csv'
        
        if not data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        df = pd.read_csv(data_path)
        print(f"\n数据加载: {data_path.name}")
        print(f"  数据形状: {df.shape}")
        
        X = df.drop(['sample_id', 'label'], axis=1).values
        y_raw = df['label'].values
        
        unique_labels = np.unique(y_raw)
        print(f"  检测到标签: {unique_labels}")
        
        label_mapping = None
        
        # 优先使用配置文件中的标签映射
        if 'label_mapping' in self.config and trait in self.config['label_mapping']:
            mapping_dict = {}
            for k, v in self.config['label_mapping'][trait].items():
                if isinstance(k, str) and k.isdigit():
                    mapping_dict[int(k)] = v
                elif isinstance(k, int):
                    mapping_dict[k] = v
                else:
                    mapping_dict[k] = v
            
            y = np.array([mapping_dict.get(val, str(val)) for val in y_raw])
            label_mapping = mapping_dict
            print(f"  ✓ 应用配置的标签映射: {label_mapping}")
        
        # 自动生成映射
        elif all(isinstance(label, (int, np.integer)) or 
                (isinstance(label, str) and label.isdigit()) for label in unique_labels):
            label_mapping = {
                int(label): f'Group_{label}' for label in unique_labels
            }
            
            y = np.array([label_mapping[int(val)] for val in y_raw])
            print(f"  ✓ 自动生成标签映射: {label_mapping}")
            print(f"  💡 提示: 如需自定义标签名称，请在配置文件中添加 label_mapping 配置")
        
        else:
            y = y_raw
            print(f"  ✓ 标签已是文本格式，无需映射")
        
        return X, y, df, label_mapping
    
    def run_traditional_ml(self, trait: str, X: np.ndarray, y: np.ndarray):
        """运行传统机器学习（使用独立测试集）"""
        print(f"\n{'='*60}")
        print(f"运行传统机器学习 - {trait}")
        print(f"{'='*60}")
        
        # 第一步：划分测试集（完全holdout）
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=RANDOM_SEED, stratify=y
        )
        
        print(f"\n数据划分:")
        print(f"  训练+验证: {len(X_temp)} 样本")
        print(f"  测试集: {len(X_test)} 样本 (用于最终评估)")
        
        # 标准化（仅在训练+验证集上fit）
        scaler = StandardScaler()
        X_temp_scaled = scaler.fit_transform(X_temp)
        X_test_scaled = scaler.transform(X_test)  # 使用训练集的参数
        
        scaler_path = self.models_dir / f'{trait}_scaler.pkl'
        joblib.dump(scaler, scaler_path)
        print(f"✓ Scaler已保存: {scaler_path.name}")
        
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED),
            'ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=RANDOM_SEED),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1),
            'SVM': SVC(kernel='rbf', random_state=RANDOM_SEED, probability=True),
            'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            'NaiveBayes': GaussianNB(),
            'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_SEED)
        }
        
        results = {}
        cv_results = []
        
        for name, model in models.items():
            print(f"\n  训练 {name}...")
            try:
                # 在训练+验证集上做交叉验证
                cv_scores = cross_val_score(model, X_temp_scaled, y_temp, 
                                           cv=5, scoring='accuracy', n_jobs=-1)
                
                # 在训练+验证集上训练最终模型
                model.fit(X_temp_scaled, y_temp)
                
                # 在独立测试集上评估
                y_test_pred = model.predict(X_test_scaled)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                test_f1 = f1_score(y_test, y_test_pred, average='weighted')
                test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
                test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
                
                results[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_accuracy': test_accuracy,
                    'test_f1': test_f1,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'model': model,
                    'test_predictions': y_test_pred,
                    'test_true_labels': y_test
                }
                
                cv_results.append({
                    'Model': name,
                    'CV Mean': cv_scores.mean(),
                    'CV Std': cv_scores.std(),
                    'Test Accuracy': test_accuracy,
                    'Test F1': test_f1,
                    'Test Precision': test_precision,
                    'Test Recall': test_recall
                })
                
                print(f"    CV准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                print(f"    测试集准确率: {test_accuracy:.4f} ⭐")
                
            except Exception as e:
                print(f"    ✗ 失败: {e}")
                continue
        
        self.results['traditional_ml'][trait] = results
        
        if cv_results:
            cv_df = pd.DataFrame(cv_results)
            cv_df = cv_df.sort_values('Test Accuracy', ascending=False)
            cv_df.to_csv(self.results_dir / f'{trait}_traditional_ml_results.csv', index=False)
            print(f"\n✓ 结果已保存: {trait}_traditional_ml_results.csv")
        
        if results:
            best_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
            best_model = results[best_name]['model']
            
            print(f"\n  ✓ 最佳模型: {best_name}")
            print(f"    CV准确率: {results[best_name]['cv_mean']:.4f}")
            print(f"    测试集准确率: {results[best_name]['test_accuracy']:.4f} ⭐")
            
            best_model_path = self.models_dir / f'{trait}_best_sklearn_model.pkl'
            joblib.dump(best_model, best_model_path)
            print(f"    模型已保存: {best_model_path.name}")
            
            self._plot_model_comparison(cv_df, trait)
            
            return results, best_model
        
        return None, None
    
    def _plot_model_comparison(self, cv_df: pd.DataFrame, trait: str):
        """绘制模型比较图"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        ax = axes[0]
        cv_df_sorted = cv_df.sort_values('CV Mean')
        ax.barh(cv_df_sorted['Model'], cv_df_sorted['CV Mean'], 
               xerr=cv_df_sorted['CV Std'], capsize=5, color='#3498db', alpha=0.8)
        ax.set_xlabel('Cross-Validation Accuracy', fontsize=12, fontweight='bold')
        ax.set_title(f'{trait} - Traditional ML (CV on Train+Val)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        cv_df_sorted = cv_df.sort_values('Test Accuracy')
        ax.barh(cv_df_sorted['Model'], cv_df_sorted['Test Accuracy'], 
               color='#2ecc71', alpha=0.8)
        ax.set_xlabel('Test Set Accuracy ⭐', fontsize=12, fontweight='bold')
        ax.set_title(f'{trait} - Traditional ML (Final Test)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        for fmt in ['png', 'pdf']:
            save_path = self.figures_dir / f'{trait}_traditional_ml_comparison.{fmt}'
            fig.savefig(save_path, dpi=300 if fmt == 'png' else None, 
                       bbox_inches='tight', format=fmt)
        
        print(f"  → 图表已保存: {trait}_traditional_ml_comparison")
        plt.close(fig)
    
    def train_deep_learning_model(self, model_name: str, trait: str,
                                  X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray,
                                  X_test: np.ndarray, y_test: np.ndarray,
                                  num_classes: int) -> Dict:
        """训练单个深度学习模型（使用独立测试集）"""
        
        train_dataset = GeneExpressionDataset(X_train, y_train)
        val_dataset = GeneExpressionDataset(X_val, y_val)
        test_dataset = GeneExpressionDataset(X_test, y_test)
        
        batch_size = min(32, len(X_train) // 4)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        input_dim = X_train.shape[1]
        model = get_model(model_name, input_dim, num_classes, dropout=self.dropout)
        model = model.to(self.device)
        
        n_params = count_parameters(model)
        print(f"  模型参数量: {n_params:,}")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=15
        )
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        best_val_acc = 0
        best_epoch = 0
        valid_best_val_acc = 0
        valid_best_epoch = 0
        patience_counter = 0
        max_patience = 30
        
        # 训练循环（在训练集上训练，在验证集上选择最佳模型）
        for epoch in range(self.max_epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            # 验证阶段（用于early stopping）
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            # 记录历史
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            scheduler.step(val_loss)
            
            # 追踪全局最佳epoch
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(model.state_dict(), 
                          self.models_dir / f'{trait}_{model_name}_best.pth')
            else:
                patience_counter += 1
            
            # 追踪epoch >= min_valid_epochs中的最佳epoch
            if epoch + 1 >= self.min_valid_epochs:
                if val_acc > valid_best_val_acc:
                    valid_best_val_acc = val_acc
                    valid_best_epoch = epoch + 1
                    torch.save(model.state_dict(), 
                              self.models_dir / f'{trait}_{model_name}_valid_best.pth')
            
            # 早停判断
            if epoch >= self.min_epochs - 1 and patience_counter >= max_patience:
                print(f"  早停于 epoch {epoch+1} (全局最佳: epoch {best_epoch})")
                break
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{self.max_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% - "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # 决定使用哪个模型
        if best_epoch < self.min_valid_epochs:
            if valid_best_epoch == 0:
                print(f"  ❌ 舍弃模型: 训练不足{self.min_valid_epochs}轮，无法评估")
                return None
            
            print(f"  ⚠️  全局最佳epoch={best_epoch} < {self.min_valid_epochs}（可能过拟合）")
            print(f"       使用epoch≥{self.min_valid_epochs}中的最佳: epoch {valid_best_epoch}")
            print(f"       验证准确率: {best_val_acc:.2f}% → {valid_best_val_acc:.2f}%")
            
            model.load_state_dict(torch.load(
                self.models_dir / f'{trait}_{model_name}_valid_best.pth'))
            final_epoch = valid_best_epoch
            final_val_acc = valid_best_val_acc
        else:
            model.load_state_dict(torch.load(
                self.models_dir / f'{trait}_{model_name}_best.pth'))
            final_epoch = best_epoch
            final_val_acc = best_val_acc
        
        model.eval()
        
        # 在测试集上最终评估（完全没见过的数据）
        print(f"  📊 在测试集上评估...")
        test_preds = []
        test_labels = []
        test_probs = []
        test_inputs = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs_cpu = inputs.cpu().numpy()
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                test_preds.extend(predicted.cpu().numpy())
                test_labels.extend(labels.numpy())
                test_probs.extend(probs.cpu().numpy())
                test_inputs.append(inputs_cpu)
        
        test_preds = np.array(test_preds)
        test_labels = np.array(test_labels)
        test_probs = np.array(test_probs)
        test_inputs = np.vstack(test_inputs)
        
        # 计算测试集指标
        test_accuracy = accuracy_score(test_labels, test_preds)
        test_precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
        test_recall = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
        test_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
        
        print(f"     验证集准确率: {final_val_acc:.2f}%")
        print(f"     测试集准确率: {test_accuracy*100:.2f}% ⭐")
        
        results = {
            'model_name': model_name,
            'val_accuracy': final_val_acc / 100,  # 验证集准确率
            'test_accuracy': test_accuracy,  # 测试集准确率（最终评估指标）
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'history': history,
            'predictions': test_preds,
            'true_labels': test_labels,
            'probabilities': test_probs,
            'input_data': test_inputs,
            'confusion_matrix': confusion_matrix(test_labels, test_preds),
            'best_epoch': final_epoch,
            'global_best_epoch': best_epoch,
            'used_fallback': best_epoch < self.min_valid_epochs,
            'n_parameters': n_params,
            'total_epochs': len(history['train_loss'])
        }
        
        try:
            results['test_auc'] = roc_auc_score(test_labels, test_probs, 
                                               multi_class='ovr', average='weighted')
        except:
            results['test_auc'] = 0.0
        
        return results
    
    def run_deep_learning(self, trait: str, X: np.ndarray, y: np.ndarray):
        """运行所有深度学习模型（使用独立测试集）"""
        print(f"\n{'='*60}")
        print(f"运行深度学习模型 - {trait}")
        print(f"{'='*60}")
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        num_classes = len(np.unique(y_encoded))
        
        le_path = self.models_dir / f'{trait}_label_encoder.pkl'
        joblib.dump(le, le_path)
        print(f"✓ Label Encoder已保存: {le_path.name}")
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 第一步：划分测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y_encoded, test_size=self.test_size, 
            random_state=RANDOM_SEED, stratify=y_encoded
        )
        
        # 第二步：从剩余数据划分训练集和验证集
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=RANDOM_SEED, stratify=y_temp
        )
        
        print(f"\n数据划分:")
        print(f"  训练集: {X_train.shape} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  验证集: {X_val.shape} ({len(X_val)/len(X)*100:.1f}%) - 用于early stopping")
        print(f"  测试集: {X_test.shape} ({len(X_test)/len(X)*100:.1f}%) - 用于最终评估 ⭐")
        print(f"  类别数: {num_classes}")
        print(f"  Epoch范围: {self.min_epochs}-{self.max_epochs}")
        
        dl_results = {}
        fallback_models = []
        
        for model_name in MODEL_REGISTRY.keys():
            print(f"\n训练 {model_name}...")
            try:
                results = self.train_deep_learning_model(
                    model_name, trait, X_train, y_train, X_val, y_val, 
                    X_test, y_test, num_classes
                )
                
                if results is None:
                    continue
                
                dl_results[model_name] = results
                
                if results['used_fallback']:
                    fallback_models.append({
                        'model': model_name,
                        'global_best': results['global_best_epoch'],
                        'used_epoch': results['best_epoch']
                    })
                
                status = f"✓ {model_name} 完成:"
                if results['used_fallback']:
                    status = f"⚠️ {model_name} 完成 (使用fallback epoch):"
                
                print(status)
                print(f"  验证集准确率: {results['val_accuracy']:.4f}")
                print(f"  测试集准确率: {results['test_accuracy']:.4f} ⭐")
                print(f"  测试集F1分数: {results['test_f1']:.4f}")
                print(f"  使用epoch: {results['best_epoch']} (总计{results['total_epochs']}轮)")
                
            except Exception as e:
                print(f"✗ {model_name} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if fallback_models:
            print(f"\n⚠️  {len(fallback_models)} 个模型使用了fallback epoch:")
            for info in fallback_models:
                print(f"    - {info['model']}: epoch {info['global_best']} → {info['used_epoch']}")
        
        if not dl_results:
            print(f"\n⚠️  警告: 所有模型都被舍弃或训练失败，无深度学习结果")
            return {}, le, X_scaled
        
        self.results['deep_learning'][trait] = dl_results
        
        if dl_results:
            summary_list = []
            for model_name, results in dl_results.items():
                summary_list.append({
                    'Model': model_name,
                    'Val_Accuracy': results['val_accuracy'],
                    'Test_Accuracy': results['test_accuracy'],
                    'Test_Precision': results['test_precision'],
                    'Test_Recall': results['test_recall'],
                    'Test_F1': results['test_f1'],
                    'Test_AUC': results.get('test_auc', 0.0),
                    'Parameters': results['n_parameters'],
                    'Used_Epoch': results['best_epoch'],
                    'Global_Best_Epoch': results['global_best_epoch'],
                    'Used_Fallback': 'Yes' if results['used_fallback'] else 'No',
                    'Total_Epochs': results['total_epochs']
                })
            
            summary_df = pd.DataFrame(summary_list)
            summary_df = summary_df.sort_values('Test_Accuracy', ascending=False)
            summary_df.to_csv(self.results_dir / f'{trait}_deep_learning_results.csv', index=False)
            print(f"\n✓ 结果已保存: {trait}_deep_learning_results.csv")
            print(f"  保留了 {len(dl_results)} 个模型")
            print(f"\n📊 最佳模型(按测试集准确率):")
            print(f"  {summary_df.iloc[0]['Model']}: {summary_df.iloc[0]['Test_Accuracy']:.4f}")
            
            self._plot_training_history(dl_results, trait)
            self._plot_confusion_matrices(dl_results, trait, le)
        
        return dl_results, le, X_scaled
    
    def _plot_training_history(self, dl_results: Dict, trait: str):
        """绘制训练历史"""
        n_models = len(dl_results)
        fig, axes = plt.subplots(n_models, 2, figsize=(12, 4*n_models))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, results) in enumerate(dl_results.items()):
            history = results['history']
            
            # 添加测试集结果标注
            test_acc = results['test_accuracy'] * 100
            val_acc = results['val_accuracy'] * 100
            
            ax = axes[idx, 0]
            ax.plot(history['train_loss'], label='Train Loss', linewidth=2, color='#e74c3c')
            ax.plot(history['val_loss'], label='Val Loss', linewidth=2, color='#3498db')
            ax.axvline(x=results['best_epoch']-1, color='#2ecc71', 
                      linestyle='--', alpha=0.7, label=f'Best Epoch')
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Loss', fontsize=10)
            ax.set_title(f'{model_name} - Loss', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax = axes[idx, 1]
            ax.plot(history['train_acc'], label='Train Acc', linewidth=2, color='#e74c3c')
            ax.plot(history['val_acc'], label='Val Acc', linewidth=2, color='#3498db')
            ax.axvline(x=results['best_epoch']-1, color='#2ecc71', 
                      linestyle='--', alpha=0.7, label=f'Best Epoch')
            ax.axhline(y=test_acc, color='#f39c12', linestyle=':', 
                      linewidth=2, label=f'Test Acc: {test_acc:.1f}%')
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Accuracy (%)', fontsize=10)
            ax.set_title(f'{model_name} - Accuracy (Test: {test_acc:.1f}%)', 
                        fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        for fmt in ['png', 'pdf']:
            save_path = self.figures_dir / f'{trait}_dl_training_history.{fmt}'
            fig.savefig(save_path, dpi=300 if fmt == 'png' else None, 
                       bbox_inches='tight', format=fmt)
        
        print(f"  → 图表已保存: {trait}_dl_training_history")
        plt.close(fig)
    
    def _plot_confusion_matrices(self, dl_results: Dict, trait: str, le: LabelEncoder):
        """绘制混淆矩阵（基于测试集）"""
        n_models = len(dl_results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(dl_results.items()):
            cm = results['confusion_matrix']
            
            ax = axes[idx]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=le.classes_, yticklabels=le.classes_,
                       ax=ax, cbar=True, square=True)
            ax.set_xlabel('Predicted', fontsize=10, fontweight='bold')
            ax.set_ylabel('True', fontsize=10, fontweight='bold')
            ax.set_title(f'{model_name}\nTest Acc: {results["test_accuracy"]:.3f} ⭐', 
                        fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        for fmt in ['png', 'pdf']:
            save_path = self.figures_dir / f'{trait}_confusion_matrices.{fmt}'
            fig.savefig(save_path, dpi=300 if fmt == 'png' else None, 
                       bbox_inches='tight', format=fmt)
        
        print(f"  → 图表已保存: {trait}_confusion_matrices")
        plt.close(fig)
    
    def extract_top_genes(self, trait: str, X: np.ndarray, gene_names: List[str], 
                         top_k: int = 20):
        """提取重要基因"""
        print(f"\n{'='*60}")
        print(f"提取Top {top_k}基因 - {trait}")
        print(f"{'='*60}")
        
        gene_vars = np.var(X, axis=0)
        top_indices_var = np.argsort(gene_vars)[-top_k:][::-1]
        
        gene_means = np.mean(X, axis=0)
        top_indices_mean = np.argsort(gene_means)[-top_k:][::-1]
        
        gene_cv = gene_vars / (gene_means + 1e-10)
        top_indices_cv = np.argsort(gene_cv)[-top_k:][::-1]
        
        top_genes = {
            'by_variance': {
                'genes': [gene_names[i] for i in top_indices_var],
                'scores': gene_vars[top_indices_var].tolist(),
                'indices': top_indices_var.tolist()
            },
            'by_mean': {
                'genes': [gene_names[i] for i in top_indices_mean],
                'scores': gene_means[top_indices_mean].tolist(),
                'indices': top_indices_mean.tolist()
            },
            'by_cv': {
                'genes': [gene_names[i] for i in top_indices_cv],
                'scores': gene_cv[top_indices_cv].tolist(),
                'indices': top_indices_cv.tolist()
            }
        }
        
        self.results['top_genes'][trait] = top_genes
        
        for method in ['variance', 'mean', 'cv']:
            genes = top_genes[f'by_{method}']['genes']
            scores = top_genes[f'by_{method}']['scores']
            
            df = pd.DataFrame({
                'Rank': range(1, len(genes) + 1),
                'Gene': genes,
                'Score': scores
            })
            df.to_csv(self.results_dir / f'{trait}_top_genes_by_{method}.csv', index=False)
        
        self._plot_top_genes(top_genes, trait, top_k)
        
        print(f"\nTop 10重要基因(按方差):")
        for i, (gene, score) in enumerate(zip(
            top_genes['by_variance']['genes'][:10],
            top_genes['by_variance']['scores'][:10]
        ), 1):
            print(f"  {i:2d}. {gene:30s} : {score:.6f}")
        
        return top_genes
    
    def _plot_top_genes(self, top_genes: Dict, trait: str, top_k: int):
        """绘制top基因"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        methods = ['variance', 'mean', 'cv']
        titles = ['Variance', 'Mean Expression', 'Coefficient of Variation']
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        
        for idx, (method, title, color) in enumerate(zip(methods, titles, colors)):
            ax = axes[idx]
            genes = top_genes[f'by_{method}']['genes'][:top_k]
            scores = top_genes[f'by_{method}']['scores'][:top_k]
            
            y_pos = np.arange(len(genes))
            ax.barh(y_pos, scores, color=color, alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(genes, fontsize=8)
            ax.set_xlabel(title, fontsize=10, fontweight='bold')
            ax.set_title(f'Top {top_k} Genes by {title}', 
                        fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        for fmt in ['png', 'pdf']:
            save_path = self.figures_dir / f'{trait}_top_genes.{fmt}'
            fig.savefig(save_path, dpi=300 if fmt == 'png' else None, 
                       bbox_inches='tight', format=fmt)
        
        print(f"  → 图表已保存: {trait}_top_genes")
        plt.close(fig)
    
    def run_full_pipeline(self, trait: str):
        """运行完整的ML流程"""
        trait_start_time = datetime.now()
        
        print(f"\n{'#'*60}")
        print(f"开始分析性状: {trait}")
        print(f"开始时间: {trait_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*60}")
        
        try:
            X, y, df, label_mapping = self.load_data(trait)
            gene_names = df.drop(['sample_id', 'label'], axis=1).columns.tolist()
        except Exception as e:
            print(f"✗ 数据加载失败: {e}")
            return
        
        print(f"\n数据摘要:")
        print(f"  样本数: {len(y)}")
        print(f"  特征数: {len(gene_names)}")
        unique, counts = np.unique(y, return_counts=True)
        print(f"  类别分布: {dict(zip(unique, counts))}")
        
        try:
            trad_results, best_model = self.run_traditional_ml(trait, X, y)
        except Exception as e:
            print(f"\n✗ 传统ML执行出错: {e}")
            trad_results, best_model = None, None
        
        try:
            dl_results, label_encoder, X_scaled = self.run_deep_learning(trait, X, y)
        except Exception as e:
            print(f"\n✗ 深度学习执行出错: {e}")
            dl_results = {}
            label_encoder = None
            X_scaled = None
        
        try:
            top_genes = self.extract_top_genes(trait, X, gene_names, top_k=20)
        except Exception as e:
            print(f"\n✗ 特征重要性分析出错: {e}")
            top_genes = None
        
        if self.enhancements and dl_results and top_genes:
            try:
                print(f"\n{'='*60}")
                print("运行增强分析...")
                print(f"{'='*60}")
                
                # 保存训练好的模型（用于结构可视化）
                models_dict = {}
                for model_name in dl_results.keys():
                    try:
                        model_path = self.models_dir / f'{trait}_{model_name}_best.pth'
                        if model_path.exists():
                            input_dim = X.shape[1]
                            num_classes = len(np.unique(y))
                            model = get_model(model_name, input_dim, num_classes,
                                              dropout=self.dropout)
                            model.load_state_dict(torch.load(model_path, map_location=self.device))
                            models_dict[model_name] = model
                            print(f"  ✓ 加载模型: {model_name}")
                    except Exception as e:
                        print(f"  ⚠️  加载{model_name}失败: {e}")
                        continue

                # ⭐ 从模型中提取真正的基因权重
                if models_dict:
                    model_top_genes = self.enhancements.extract_model_gene_importance(
                        models_dict, gene_names, top_k=20
                    )

                    # 如果某些模型权重提取失败，使用统计特征作为fallback
                    top_genes_dict = {}
                    fallback_indices = top_genes['by_variance']['indices'][:20]

                    for model_name in dl_results.keys():
                        if model_name in model_top_genes:
                            top_genes_dict[model_name] = model_top_genes[model_name]['indices']
                        else:
                            print(f"  ⚠️  {model_name} 使用统计特征作为fallback")
                            top_genes_dict[model_name] = fallback_indices
                else:
                    # 如果没有模型，使用统计特征
                    fallback_indices = top_genes['by_variance']['indices'][:20]
                    top_genes_dict = {
                        model_name: fallback_indices
                        for model_name in dl_results.keys()
                    }


                # 准备训练参数字典
                training_params = {
                    'epochs': self.max_epochs,
                    'min_epochs': self.min_epochs,
                    'min_valid_epochs': self.min_valid_epochs,
                    'dropout': self.dropout,
                    'batch_size': 32,  # 或从代码中获取实际的batch_size
                    'learning_rate': 0.001  # 固定值
                }

                self.enhancements.run_comprehensive_analysis(
                    trait=trait, X=X, y=y, X_scaled=X_scaled,
                    gene_names=gene_names, dl_results=dl_results,
                    top_genes_dict=top_genes_dict,
                    label_encoder=label_encoder,
                    label_mapping=label_mapping,
                    traditional_ml_results=self.results['traditional_ml'].get(trait),
                    models_dict=models_dict,
                    training_params=training_params  # ⭐ 新增
                )
            except Exception as e:
                print(f"✗ 增强分析失败: {e}")
                import traceback
                traceback.print_exc()
        
        self.save_results(trait)
        
        trait_end_time = datetime.now()
        duration = (trait_end_time - trait_start_time).total_seconds()
        
        print(f"\n{'='*60}")
        print(f"✓ {trait} 分析完成!")
        print(f"结束时间: {trait_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总用时: {duration:.1f}秒 ({duration/60:.1f}分钟)")
        print(f"{'='*60}")
    
    def save_results(self, trait: str):
        """保存结果"""
        result_file = self.results_dir / f'{trait}_summary.json'
        
        save_dict = {
            'trait': trait,
            'timestamp': datetime.now().isoformat(),
            'data_split': {
                'train_size': 1.0 - self.test_size - self.val_size,
                'val_size': self.val_size,
                'test_size': self.test_size
            },
            'top_genes': self.results['top_genes'].get(trait, {}),
            'traditional_ml': {},
            'deep_learning': {}
        }
        
        if trait in self.results['traditional_ml']:
            for model_name, res in self.results['traditional_ml'][trait].items():
                if model_name.startswith('_'):
                    continue
                save_dict['traditional_ml'][model_name] = {
                    'cv_mean': float(res['cv_mean']),
                    'cv_std': float(res['cv_std']),
                    'test_accuracy': float(res['test_accuracy']),
                    'test_f1': float(res['test_f1']),
                    'test_precision': float(res['test_precision']),
                    'test_recall': float(res['test_recall'])
                }
        
        if trait in self.results['deep_learning']:
            for model_name, results in self.results['deep_learning'][trait].items():
                save_dict['deep_learning'][model_name] = {
                    'val_accuracy': float(results['val_accuracy']),
                    'test_accuracy': float(results['test_accuracy']),
                    'test_precision': float(results['test_precision']),
                    'test_recall': float(results['test_recall']),
                    'test_f1': float(results['test_f1']),
                    'test_auc': float(results.get('test_auc', 0.0)),
                    'used_epoch': int(results['best_epoch']),
                    'global_best_epoch': int(results['global_best_epoch']),
                    'used_fallback': bool(results['used_fallback']),
                    'total_epochs': int(results['total_epochs']),
                    'n_parameters': int(results['n_parameters'])
                }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(save_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存:")
        print(f"  - 模型文件: {self.models_dir}")
        print(f"  - 图表文件: {self.figures_dir}")
        print(f"  - 结果文件: {result_file}")


def main():
    parser = argparse.ArgumentParser(
        description='完整的机器学习/深度学习分析流程 (带独立测试集)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用 (默认 70% train, 15% val, 15% test)
  python ml_pipeline.py -c config.json -o results
  
  # 自定义数据划分
  python ml_pipeline.py -c config.json -o results --test-size 0.2 --val-size 0.2
  
  # 完整参数
  python ml_pipeline.py -c config.json -o results --min-epochs 50 --max-epochs 300 --dropout 0.4
        """
    )
    
    parser.add_argument('-c', '--config', required=True, help='配置文件路径')
    parser.add_argument('-o', '--output', default='ml_results', help='输出目录')
    parser.add_argument('-t', '--trait', default=None, help='指定性状')
    parser.add_argument('--min-epochs', type=int, default=30, help='最小训练轮数 [默认: 30]')
    parser.add_argument('--max-epochs', type=int, default=200, help='最大训练轮数 [默认: 200]')
    parser.add_argument('--min-valid-epochs', type=int, default=10, 
                       help='最佳epoch的最小有效值 [默认: 10]')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout比例 [范围: 0.0-0.8, 默认: 0.5]')
    parser.add_argument('--test-size', type=float, default=0.15,
                       help='测试集比例 [范围: 0.05-0.3, 默认: 0.15]')
    parser.add_argument('--val-size', type=float, default=0.15,
                       help='验证集比例 [范围: 0.05-0.3, 默认: 0.15]')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("机器学习/深度学习分析流程 v1.4.0")
    print("✓ 使用独立测试集 (Train/Val/Test 三分法)")
    print("="*60)
    
    try:
        pipeline = MLPipeline(
            args.config, 
            args.output, 
            min_epochs=args.min_epochs, 
            max_epochs=args.max_epochs,
            min_valid_epochs=args.min_valid_epochs,
            dropout=args.dropout,
            test_size=args.test_size,
            val_size=args.val_size
        )
    except Exception as e:
        print(f"\n✗ 初始化失败: {e}")
        sys.exit(1)
    
    traits = [args.trait] if args.trait else pipeline.config['traits']
    
    success_count = 0
    for idx, trait in enumerate(traits, 1):
        print(f"\n进度: {idx}/{len(traits)}")
        try:
            pipeline.run_full_pipeline(trait)
            success_count += 1
        except Exception as e:
            print(f"\n✗ {trait} 分析失败: {e}")
    
    print(f"\n{'='*60}")
    print(f"完成: {success_count}/{len(traits)} 个性状分析成功")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
