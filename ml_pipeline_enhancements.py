"""
机器学习增强分析模块 - 完整最终版 (修复版)
修复了标签处理问题：y直接是字符串标签，不需要转换为索引
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import networkx as nx
from scipy import stats
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')


class MLPipelineEnhancements:
    """增强分析功能（完整最终版）"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / 'figures'
        self.enhanced_dir = self.figures_dir / 'enhanced'
        self.enhanced_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.output_dir / 'results'
        
        print(f"✓ 增强分析目录: {self.enhanced_dir}")
    
    def extract_model_gene_importance(self, models_dict: Dict, gene_names: List[str], 
                                     top_k: int = 20) -> Dict:
        """从训练好的模型中提取基因重要性
        
        Args:
            models_dict: {model_name: trained_model}
            gene_names: 基因名列表
            top_k: 返回top K个基因
            
        Returns:
            {model_name: {'genes': [...], 'scores': [...], 'indices': [...]}}
        """
        import torch.nn as nn
        
        print(f"\n  提取模型基因权重 (Top {top_k})...")
        
        model_top_genes = {}
        
        for model_name, model in models_dict.items():
            try:
                model.eval()
                
                # 找到第一个Linear层（输入层→隐藏层）
                first_linear = None
                for module in model.modules():
                    if isinstance(module, nn.Linear):
                        first_linear = module
                        break
                
                if first_linear is None:
                    print(f"    ⚠️  {model_name}: 未找到Linear层，使用统计特征")
                    continue
                
                # 提取权重矩阵 [out_features, in_features]
                weight_matrix = first_linear.weight.data.cpu().numpy()
                
                # 计算每个基因的重要性（L2范数）
                gene_importance = np.linalg.norm(weight_matrix, axis=0)
                
                # 排序获取top基因
                top_indices = np.argsort(gene_importance)[-top_k:][::-1]
                top_genes = [gene_names[i] for i in top_indices]
                top_scores = gene_importance[top_indices].tolist()
                
                model_top_genes[model_name] = {
                    'genes': top_genes,
                    'scores': top_scores,
                    'indices': top_indices.tolist()
                }
                
                print(f"    ✓ {model_name}: 提取到 {len(top_genes)} 个重要基因")
                print(f"      Top 3: {', '.join(top_genes[:3])}")
                
            except Exception as e:
                print(f"    ✗ {model_name} 权重提取失败: {e}")
                continue
        
        return model_top_genes
    
    def _get_metric(self, results: Dict, metric: str, default=0.0):
        """安全获取指标值"""
        test_key = f'test_{metric}'
        if test_key in results:
            return results[test_key]
        if metric in results:
            return results[metric]
        return default
    
    def run_comprehensive_analysis(self, trait: str, X: np.ndarray, y: np.ndarray,
                                   X_scaled: np.ndarray, gene_names: List[str],
                                   dl_results: Dict, top_genes_dict: Dict,
                                   label_encoder, label_mapping: Optional[Dict] = None,
                                   traditional_ml_results: Optional[Dict] = None,
                                   models_dict: Optional[Dict] = None,
                                   training_params: Optional[Dict] = None):
        """运行综合增强分析"""
        
        print("\n  执行增强分析 (保存到 enhanced/ 目录)...")
        
        if training_params:
            print(f"  训练参数: Epochs={training_params.get('epochs', 'N/A')}, "
                  f"Dropout={training_params.get('dropout', 'N/A')}, "
                  f"Batch={training_params.get('batch_size', 'N/A')}")
        
        try:
            # 1. 传统ML分析
            if traditional_ml_results:
                if '_lasso_info' in traditional_ml_results:
                    self._plot_lasso_path(traditional_ml_results['_lasso_info'], trait)
                if '_rf_training_curve' in traditional_ml_results:
                    self._plot_rf_training_curve(traditional_ml_results['_rf_training_curve'], trait)
            
            # 2. 深度学习综合分析
            if dl_results:
                self._plot_dl_performance_comparison(dl_results, trait, training_params)
                self._plot_precision_recall_tradeoff(dl_results, trait)
                self._plot_prediction_distribution(dl_results, trait, label_encoder)
                self._plot_learning_curves_comparison(dl_results, trait, training_params)
                self._plot_model_complexity_vs_performance(dl_results, trait)
                self._plot_loss_function_analysis(dl_results, trait, training_params)
            
            # 3. 特征分析
            if top_genes_dict and dl_results:
                self._plot_feature_importance_heatmap(
                    X_scaled, y, gene_names, top_genes_dict, trait, label_encoder
                )
                self._plot_gene_correlation_matrix(
                    X_scaled, gene_names, top_genes_dict, trait
                )
                
                # 为每个模型生成基因相互作用网络
                print("  生成基因相互作用网络（每个模型独立）...")
                all_network_genes = {}
                for model_name in top_genes_dict.keys():
                    network_genes = self._plot_gene_interaction_network(
                        X_scaled, gene_names, top_genes_dict, trait, model_name
                    )
                    all_network_genes[model_name] = network_genes
                
                # 网络基因箱线图 + 显著性分析
                if all_network_genes:
                    first_model_genes = list(all_network_genes.values())[0]
                    if first_model_genes:
                        self._plot_gene_boxplots_with_significance(
                            X_scaled, y, gene_names, first_model_genes, trait, label_encoder
                        )
                
                # 分组表达量显著性分析
                self._plot_gene_expression_groupwise_significance(
                    X_scaled, y, gene_names, top_genes_dict, trait, label_encoder
                )
                
                # 每个模型Top10基因的权重箱线图
                self._plot_top_gene_importance_boxplots(
                    X_scaled, y, gene_names, top_genes_dict, trait, label_encoder
                )
            
            # 4. 降维可视化
            self._plot_dimensionality_reduction(
                X_scaled, y, trait, label_encoder, label_mapping
            )
            
            # 5. 真实vs预测散点图
            if dl_results:
                self._plot_true_vs_predicted_scatter(
                    dl_results, X_scaled, y, trait, label_encoder
                )
            
            # 6. 神经网络结构图
            if models_dict:
                self._plot_neural_network_architectures(models_dict, trait, training_params)
            
            print("  ✓ 增强分析完成")
            
        except Exception as e:
            print(f"  ⚠️  部分增强分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_lasso_path(self, lasso_info: Dict, trait: str):
        """Lasso正则化路径"""
        if not lasso_info or 'coef_path' not in lasso_info:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        coef_path = lasso_info['coef_path']
        alphas = lasso_info['alphas']
        optimal_alpha = lasso_info['optimal_alpha']
        
        ax = axes[0]
        nonzero_coef = np.any(coef_path != 0, axis=1)
        for i in range(coef_path.shape[0]):
            if nonzero_coef[i]:
                ax.plot(np.log10(alphas), coef_path[i, :], alpha=0.6)
        ax.axvline(x=np.log10(optimal_alpha), color='red', linestyle='--', 
                   linewidth=2, label=f'Optimal α={optimal_alpha:.4f}')
        ax.set_xlabel('Log(α)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Coefficient Value', fontsize=12, fontweight='bold')
        ax.set_title('Lasso Regularization Path', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        if 'cv_mse' in lasso_info:
            cv_mse = lasso_info['cv_mse']
            ax.plot(np.log10(alphas), cv_mse, 'b-', linewidth=2)
            ax.axvline(x=np.log10(optimal_alpha), color='red', linestyle='--', 
                      linewidth=2, label=f'Optimal α={optimal_alpha:.4f}')
            ax.set_xlabel('Log(α)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
            ax.set_title('Cross-Validation Error', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{trait} - Lasso Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        for fmt in ['png', 'pdf']:
            save_path = self.enhanced_dir / f'{trait}_lasso_path.{fmt}'
            fig.savefig(save_path, dpi=300 if fmt == 'png' else None, bbox_inches='tight')
        
        print(f"    → Lasso Path")
        plt.close(fig)
    
    def _plot_rf_training_curve(self, rf_curve: Dict, trait: str):
        """RandomForest训练曲线"""
        if not rf_curve:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n_trees = rf_curve['n_trees']
        train_errors = rf_curve['train_errors']
        test_errors = rf_curve['test_errors']
        cv_errors = rf_curve['cv_errors']
        
        ax.plot(n_trees, train_errors, 'o-', label='Train Error', 
               color='#e74c3c', linewidth=2, markersize=4)
        ax.plot(n_trees, test_errors, 's-', label='Test Error', 
               color='#3498db', linewidth=2, markersize=4)
        ax.plot(n_trees, cv_errors, '^-', label='CV Error', 
               color='#2ecc71', linewidth=2, markersize=4)
        
        min_cv_idx = np.argmin(cv_errors)
        ax.axvline(x=n_trees[min_cv_idx], color='gray', linestyle='--', 
                  alpha=0.5, label=f'Optimal: {n_trees[min_cv_idx]} trees')
        
        ax.set_xlabel('Number of Trees', fontsize=12, fontweight='bold')
        ax.set_ylabel('Error Rate', fontsize=12, fontweight='bold')
        ax.set_title(f'{trait} - RandomForest Training Curve', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        for fmt in ['png', 'pdf']:
            save_path = self.enhanced_dir / f'{trait}_rf_training_curve.{fmt}'
            fig.savefig(save_path, dpi=300 if fmt == 'png' else None, bbox_inches='tight')
        
        print(f"    → RF Training Curve")
        plt.close(fig)
    
    def _plot_dl_performance_comparison(self, dl_results: Dict, trait: str, 
                                        training_params: Optional[Dict] = None):
        """深度学习模型性能雷达图"""
        if not dl_results:
            return
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(dl_results)))
        
        for idx, (model_name, results) in enumerate(dl_results.items()):
            values = [self._get_metric(results, m, 0.0) for m in metrics]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, 
                   color=colors[idx], alpha=0.8)
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, size=12)
        ax.set_ylim(0, 1)
        
        title = f'{trait} - Model Performance (Test Set)'
        if training_params:
            title += f'\nEpochs={training_params.get("epochs", "N/A")}, '
            title += f'Dropout={training_params.get("dropout", "N/A")}, '
            title += f'LR={training_params.get("learning_rate", "N/A")}'
        
        ax.set_title(title, size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        
        for fmt in ['png', 'pdf']:
            save_path = self.enhanced_dir / f'{trait}_dl_radar.{fmt}'
            fig.savefig(save_path, dpi=300 if fmt == 'png' else None, bbox_inches='tight')
        
        print(f"    → Performance Radar Chart")
        plt.close(fig)
    
    def _plot_precision_recall_tradeoff(self, dl_results: Dict, trait: str):
        """Precision vs Recall - 带y=x虚线"""
        if not dl_results:
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        has_data = False
        for model_name, results in dl_results.items():
            precision = self._get_metric(results, 'precision', 0.0)
            recall = self._get_metric(results, 'recall', 0.0)
            accuracy = self._get_metric(results, 'accuracy', 0.0)
            
            if precision > 0 or recall > 0:
                has_data = True
                size = max(accuracy * 1000, 100)
                ax.scatter(recall, precision, s=size, alpha=0.6, label=model_name)
                ax.annotate(model_name, (recall, precision), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        if has_data:
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, 
                   label='Perfect Balance (y=x)', zorder=0)
            
            ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
            ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
            ax.set_title(f'{trait} - Precision vs Recall Trade-off\n(Bubble Size = Accuracy)', 
                        fontsize=16, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1.05])
            ax.set_ylim([0, 1.05])
            
            plt.tight_layout()
            
            for fmt in ['png', 'pdf']:
                save_path = self.enhanced_dir / f'{trait}_precision_recall.{fmt}'
                fig.savefig(save_path, dpi=300 if fmt == 'png' else None, bbox_inches='tight')
            
            print(f"    → Precision-Recall Trade-off")
        else:
            print(f"    ⚠️  Precision-Recall图无数据")
        
        plt.close(fig)
    
    def _plot_prediction_distribution(self, dl_results: Dict, trait: str, label_encoder):
        """预测置信度分布"""
        if not dl_results:
            return
        
        valid_models = {k: v for k, v in dl_results.items() if 'probabilities' in v}
        
        if not valid_models:
            print(f"    ⚠️  预测分布图无数据")
            return
        
        n_models = len(valid_models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(valid_models.items()):
            ax = axes[idx]
            
            probs = results['probabilities']
            true_labels = results['true_labels']
            max_probs = probs.max(axis=1)
            predictions = probs.argmax(axis=1)
            correct = (predictions == true_labels)
            
            ax.hist([max_probs[correct], max_probs[~correct]], 
                   bins=20, label=['Correct', 'Incorrect'],
                   alpha=0.7, color=['green', 'red'])
            
            ax.set_xlabel('Confidence', fontsize=11, fontweight='bold')
            ax.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax.set_title(f'{model_name}\nAcc: {self._get_metric(results, "accuracy", 0):.3f}',
                       fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{trait} - Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        for fmt in ['png', 'pdf']:
            save_path = self.enhanced_dir / f'{trait}_pred_distribution.{fmt}'
            fig.savefig(save_path, dpi=300 if fmt == 'png' else None, bbox_inches='tight')
        
        print(f"    → Prediction Distribution")
        plt.close(fig)

    def _plot_learning_curves_comparison(self, dl_results: Dict, trait: str,
                                         training_params: Optional[Dict] = None):
        """学习曲线对比"""
        if not dl_results:
            return

        valid_models = {k: v for k, v in dl_results.items() if 'history' in v}

        if not valid_models:
            print(f"    ⚠️  学习曲线无数据")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(valid_models)))

        ax = axes[0]
        for idx, (model_name, results) in enumerate(valid_models.items()):
            epochs = len(results['history']['val_loss'])
            ax.plot(range(1, epochs+1), results['history']['val_loss'],
                   label=model_name, color=colors[idx], linewidth=2, alpha=0.8)
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
        ax.set_title('Loss Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        for idx, (model_name, results) in enumerate(valid_models.items()):
            epochs = len(results['history']['val_acc'])
            ax.plot(range(1, epochs+1), results['history']['val_acc'],
                   label=model_name, color=colors[idx], linewidth=2, alpha=0.8)
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        title = f'{trait} - Learning Curves'
        if training_params:
            title += f'\n(Trained for {training_params.get("epochs", "N/A")} epochs, '
            title += f'Dropout={training_params.get("dropout", "N/A")})'

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        for fmt in ['png', 'pdf']:
            save_path = self.enhanced_dir / f'{trait}_learning_curves.{fmt}'
            fig.savefig(save_path, dpi=300 if fmt == 'png' else None, bbox_inches='tight')

        print(f"    → Learning Curves")
        plt.close(fig)

    def _plot_model_complexity_vs_performance(self, dl_results: Dict, trait: str):
        """模型复杂度vs性能"""
        if not dl_results:
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        has_data = False
        for model_name, results in dl_results.items():
            n_params = results.get('n_parameters', 0)
            accuracy = self._get_metric(results, 'accuracy', 0.0)
            f1 = self._get_metric(results, 'f1', 0.0)

            if n_params > 0 and accuracy > 0:
                has_data = True
                size = max(f1 * 1000, 100)
                ax.scatter(n_params, accuracy, s=size, alpha=0.6, label=model_name)
                ax.annotate(model_name, (n_params, accuracy),
                           xytext=(5, 5), textcoords='offset points', fontsize=10)

        if has_data:
            ax.set_xlabel('Parameters', fontsize=14, fontweight='bold')
            ax.set_ylabel('Test Accuracy', fontsize=14, fontweight='bold')
            ax.set_title(f'{trait} - Model Complexity vs Performance\n(Bubble Size = F1 Score)',
                        fontsize=16, fontweight='bold')
            ax.set_xscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            for fmt in ['png', 'pdf']:
                save_path = self.enhanced_dir / f'{trait}_complexity_performance.{fmt}'
                fig.savefig(save_path, dpi=300 if fmt == 'png' else None, bbox_inches='tight')

            print(f"    → Complexity Analysis")
        else:
            print(f"    ⚠️  复杂度分析图无数据")

        plt.close(fig)

    def _plot_loss_function_analysis(self, dl_results: Dict, trait: str,
                                     training_params: Optional[Dict] = None):
        """损失函数分析"""
        valid_models = {k: v for k, v in dl_results.items() if 'history' in v}

        if not valid_models:
            print(f"    ⚠️  损失函数分析无数据")
            return

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        colors = plt.cm.tab10(np.linspace(0, 1, len(valid_models)))

        # 1. 训练损失
        ax1 = fig.add_subplot(gs[0, 0])
        for idx, (name, res) in enumerate(valid_models.items()):
            epochs = len(res['history']['train_loss'])
            ax1.plot(range(1, epochs+1), res['history']['train_loss'],
                    label=name, color=colors[idx], linewidth=2, alpha=0.8)
        ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Train Loss', fontsize=11, fontweight='bold')
        ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. 验证损失
        ax2 = fig.add_subplot(gs[0, 1])
        for idx, (name, res) in enumerate(valid_models.items()):
            epochs = len(res['history']['val_loss'])
            ax2.plot(range(1, epochs+1), res['history']['val_loss'],
                    label=name, color=colors[idx], linewidth=2, alpha=0.8)
        ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Val Loss', fontsize=11, fontweight='bold')
        ax2.set_title('Validation Loss', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. 过拟合间隙
        ax3 = fig.add_subplot(gs[0, 2])
        for idx, (name, res) in enumerate(valid_models.items()):
            gap = np.array(res['history']['val_loss']) - np.array(res['history']['train_loss'])
            epochs = len(gap)
            ax3.plot(range(1, epochs+1), gap, label=name,
                    color=colors[idx], linewidth=2, alpha=0.8)
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Gap', fontsize=11, fontweight='bold')
        ax3.set_title('Overfitting Gap\n(Val-Train)', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 4-6. 训练准确率
        ax4 = fig.add_subplot(gs[1, 0])
        for idx, (name, res) in enumerate(valid_models.items()):
            epochs = len(res['history']['train_acc'])
            ax4.plot(range(1, epochs+1), res['history']['train_acc'],
                    label=name, color=colors[idx], linewidth=2, alpha=0.8)
        ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Train Acc (%)', fontsize=11, fontweight='bold')
        ax4.set_title('Training Accuracy', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[1, 1])
        for idx, (name, res) in enumerate(valid_models.items()):
            epochs = len(res['history']['val_acc'])
            ax5.plot(range(1, epochs+1), res['history']['val_acc'],
                    label=name, color=colors[idx], linewidth=2, alpha=0.8)
        ax5.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Val Acc (%)', fontsize=11, fontweight='bold')
        ax5.set_title('Validation Accuracy', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(gs[1, 2])
        for idx, (name, res) in enumerate(valid_models.items()):
            acc_gap = np.array(res['history']['train_acc']) - np.array(res['history']['val_acc'])
            epochs = len(acc_gap)
            ax6.plot(range(1, epochs+1), acc_gap, label=name,
                    color=colors[idx], linewidth=2, alpha=0.8)
        ax6.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax6.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Gap (%)', fontsize=11, fontweight='bold')
        ax6.set_title('Accuracy Gap\n(Train-Val)', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)

        # 7-9. 汇总统计
        ax7 = fig.add_subplot(gs[2, :])
        model_names = list(valid_models.keys())
        final_train_loss = [valid_models[m]['history']['train_loss'][-1] for m in model_names]
        final_val_loss = [valid_models[m]['history']['val_loss'][-1] for m in model_names]
        final_train_acc = [valid_models[m]['history']['train_acc'][-1] for m in model_names]
        final_val_acc = [valid_models[m]['history']['val_acc'][-1] for m in model_names]

        x = np.arange(len(model_names))
        width = 0.2

        ax7.bar(x - 1.5*width, final_train_loss, width, label='Train Loss', alpha=0.8)
        ax7.bar(x - 0.5*width, final_val_loss, width, label='Val Loss', alpha=0.8)
        ax7.bar(x + 0.5*width, [a/100 for a in final_train_acc], width, label='Train Acc/100', alpha=0.8)
        ax7.bar(x + 1.5*width, [a/100 for a in final_val_acc], width, label='Val Acc/100', alpha=0.8)

        ax7.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax7.set_title('Final Metrics Summary', fontsize=12, fontweight='bold')
        ax7.set_xticks(x)
        ax7.set_xticklabels(model_names, rotation=45, ha='right')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3, axis='y')

        title = f'{trait} - Loss Function Analysis'
        if training_params:
            title += f'\n(Epochs={training_params.get("epochs", "N/A")}, '
            title += f'Dropout={training_params.get("dropout", "N/A")}, '
            title += f'Batch Size={training_params.get("batch_size", "N/A")})'

        plt.suptitle(title, fontsize=16, fontweight='bold')

        for fmt in ['png', 'pdf']:
            save_path = self.enhanced_dir / f'{trait}_loss_analysis.{fmt}'
            fig.savefig(save_path, dpi=300 if fmt == 'png' else None, bbox_inches='tight')

        print(f"    → Loss Analysis")
        plt.close(fig)

    def _plot_feature_importance_heatmap(self, X: np.ndarray, y: np.ndarray,
                                         gene_names: List[str], top_genes_dict: Dict,
                                         trait: str, label_encoder):
        """基因表达热图 - 修复版：直接使用字符串标签"""
        if not top_genes_dict:
            return

        first_model = list(top_genes_dict.keys())[0]
        top_indices = top_genes_dict[first_model][:20]
        X_top = X[:, top_indices]
        top_gene_names = [gene_names[i] for i in top_indices]

        unique_labels = label_encoder.classes_
        heatmap_data = []
        for label in unique_labels:
            mask = (y == label)  # 🔧 直接用字符串比较
            mean_expr = X_top[mask].mean(axis=0)
            heatmap_data.append(mean_expr)

        heatmap_data = np.array(heatmap_data)

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(heatmap_data, annot=False, cmap='RdYlBu_r',
                   xticklabels=top_gene_names, yticklabels=label_encoder.classes_,
                   cbar_kws={'label': 'Mean Expression'}, ax=ax)

        ax.set_xlabel('Top 20 Genes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Class', fontsize=12, fontweight='bold')
        ax.set_title(f'{trait} - Gene Expression Heatmap', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        for fmt in ['png', 'pdf']:
            save_path = self.enhanced_dir / f'{trait}_gene_heatmap.{fmt}'
            fig.savefig(save_path, dpi=300 if fmt == 'png' else None, bbox_inches='tight')

        print(f"    → Gene Heatmap")
        plt.close(fig)

    def _plot_gene_correlation_matrix(self, X: np.ndarray, gene_names: List[str],
                                      top_genes_dict: Dict, trait: str):
        """基因相关性矩阵"""
        if not top_genes_dict:
            return

        first_model = list(top_genes_dict.keys())[0]
        top_indices = top_genes_dict[first_model][:15]
        X_top = X[:, top_indices]
        top_gene_names = [gene_names[i] for i in top_indices]
        corr_matrix = np.corrcoef(X_top.T)

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   xticklabels=top_gene_names, yticklabels=top_gene_names,
                   vmin=-1, vmax=1, center=0, ax=ax)

        ax.set_title(f'{trait} - Gene Correlation Matrix', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        for fmt in ['png', 'pdf']:
            save_path = self.enhanced_dir / f'{trait}_gene_correlation.{fmt}'
            fig.savefig(save_path, dpi=300 if fmt == 'png' else None, bbox_inches='tight')

        print(f"    → Gene Correlation Matrix")
        plt.close(fig)

    def _plot_gene_interaction_network(self, X_scaled: np.ndarray, gene_names: List[str],
                                       top_genes_dict: Dict, trait: str, model_name: str,
                                       correlation_threshold: float = 0.5):
        """基因-基因相互作用网络（Cytoscape风格）- 为单个模型生成"""
        if not top_genes_dict or model_name not in top_genes_dict:
            return []

        print(f"    → Gene Interaction Network ({model_name})...")

        top_indices = top_genes_dict[model_name][:20]
        X_top = X_scaled[:, top_indices]
        top_gene_names = [gene_names[i] for i in top_indices]

        corr_matrix = np.corrcoef(X_top.T)

        G = nx.Graph()

        for gene in top_gene_names:
            G.add_node(gene)

        for i in range(len(top_gene_names)):
            for j in range(i+1, len(top_gene_names)):
                corr = abs(corr_matrix[i, j])
                if corr > correlation_threshold:
                    G.add_edge(top_gene_names[i], top_gene_names[j], weight=corr)

        if len(G.edges()) == 0:
            print(f"      ⚠️  无显著相关性（|r| > {correlation_threshold}），降低阈值")
            correlation_threshold = 0.3
            for i in range(len(top_gene_names)):
                for j in range(i+1, len(top_gene_names)):
                    corr = abs(corr_matrix[i, j])
                    if corr > correlation_threshold:
                        G.add_edge(top_gene_names[i], top_gene_names[j], weight=corr)

        fig, ax = plt.subplots(figsize=(16, 16))
        ax.set_facecolor('white')

        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        degrees = dict(G.degree())
        node_sizes = [300 + degrees.get(node, 0) * 100 for node in G.nodes()]

        node_colors = [degrees.get(node, 0) for node in G.nodes()]

        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]

        nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights],
                              alpha=0.3, edge_color='gray', ax=ax)

        nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                       node_color=node_colors, cmap='tab20',
                                       alpha=0.85, edgecolors='black',
                                       linewidths=2, ax=ax)

        labels = {node: node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10,
                               font_weight='bold', font_family='sans-serif',
                               ax=ax)

        if len(node_colors) > 0 and max(node_colors) > 0:
            sm = plt.cm.ScalarMappable(cmap='tab20',
                                       norm=plt.Normalize(vmin=min(node_colors),
                                                         vmax=max(node_colors)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Node Degree (Connections)', fontsize=12, fontweight='bold')

        ax.set_title(f'{trait} - {model_name}\nGene Interaction Network\n'
                    f'(Top {len(top_gene_names)} Genes, |r| > {correlation_threshold})\n'
                    f'Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}',
                    fontsize=16, fontweight='bold', pad=20)

        ax.axis('off')
        plt.tight_layout()

        for fmt in ['png', 'pdf']:
            save_path = self.enhanced_dir / f'{trait}_{model_name}_gene_network.{fmt}'
            fig.savefig(save_path, dpi=300 if fmt == 'png' else None,
                       bbox_inches='tight', facecolor='white')

        print(f"      ✓ {model_name} Gene Network")
        plt.close(fig)

        return list(G.nodes())

    def _plot_gene_boxplots_with_significance(self, X_scaled: np.ndarray, y: np.ndarray,
                                              gene_names: List[str], network_genes: List[str],
                                              trait: str, label_encoder):
        """网络基因箱线图 + 显著性分析 - 修复版：直接使用字符串标签"""
        if not network_genes:
            return

        print("    → Gene Expression Boxplots with Significance...")

        gene_indices = [gene_names.index(g) for g in network_genes if g in gene_names]

        if len(gene_indices) == 0:
            print("      ⚠️  无有效基因")
            return

        n_genes = min(12, len(gene_indices))
        gene_indices = gene_indices[:n_genes]
        selected_genes = [gene_names[i] for i in gene_indices]

        n_cols = 4
        n_rows = int(np.ceil(n_genes / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten() if n_genes > 1 else [axes]

        for idx, (gene_idx, gene_name) in enumerate(zip(gene_indices, selected_genes)):
            ax = axes[idx]

            gene_expr = X_scaled[:, gene_idx]
            data_for_plot = []
            labels_for_plot = []

            # 🔧 修复：直接使用字符串标签比较
            for class_label in label_encoder.classes_:
                mask = (y == class_label)  # 直接用字符串比较
                class_data = gene_expr[mask]
                data_for_plot.append(class_data)
                labels_for_plot.append(class_label)

            # 过滤空数据
            valid_indices = [i for i, d in enumerate(data_for_plot) if len(d) > 0]

            if len(valid_indices) < 2:
                empty_classes = [labels_for_plot[i] for i, d in enumerate(data_for_plot) if len(d) == 0]
                ax.text(0.5, 0.5, f'{gene_name}\n⚠️ 数据异常\n空类别: {empty_classes}',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
                ax.axis('off')
                continue

            valid_data = [data_for_plot[i] for i in valid_indices]
            valid_labels = [labels_for_plot[i] for i in valid_indices]

            bp = ax.boxplot(valid_data, labels=valid_labels, patch_artist=True,
                           showmeans=True, meanprops=dict(marker='D', markerfacecolor='red',
                                                         markeredgecolor='red', markersize=8))

            colors = plt.cm.Set3(np.linspace(0, 1, len(valid_labels)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            y_max = max([d.max() for d in valid_data])
            y_min = min([d.min() for d in valid_data])
            y_range = y_max - y_min

            sig_results = []
            for i, j in combinations(range(len(valid_data)), 2):
                if len(valid_data[i]) > 1 and len(valid_data[j]) > 1:
                    stat, pval = stats.mannwhitneyu(valid_data[i], valid_data[j],
                                                   alternative='two-sided')
                    sig_results.append((i, j, pval))

            sig_results.sort(key=lambda x: x[2])
            for rank, (i, j, pval) in enumerate(sig_results[:3]):
                if pval < 0.05:
                    if pval < 0.001:
                        sig_marker = '***'
                    elif pval < 0.01:
                        sig_marker = '**'
                    else:
                        sig_marker = '*'

                    y_pos = y_max + y_range * 0.1 * (rank + 1)
                    ax.plot([i+1, j+1], [y_pos, y_pos], 'k-', linewidth=1.5)
                    ax.text((i+j)/2+1, y_pos, sig_marker, ha='center', va='bottom',
                           fontsize=14, fontweight='bold')

            ax.set_ylabel('Expression (Scaled)', fontsize=11, fontweight='bold')
            ax.set_title(f'{gene_name}\n(Network Gene)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45)

        for idx in range(n_genes, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'{trait} - Network Gene Expression Boxplots\n'
                    f'(* p<0.05, ** p<0.01, *** p<0.001, Mann-Whitney U test)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        for fmt in ['png', 'pdf']:
            save_path = self.enhanced_dir / f'{trait}_network_gene_boxplots.{fmt}'
            fig.savefig(save_path, dpi=300 if fmt == 'png' else None, bbox_inches='tight')

        print(f"      ✓ Network Gene Boxplots")
        plt.close(fig)

    def _plot_gene_expression_groupwise_significance(self, X_scaled: np.ndarray, y: np.ndarray,
                                                     gene_names: List[str], top_genes_dict: Dict,
                                                     trait: str, label_encoder):
        """分组表达量显著性分析（热图形式）- 修复版：直接使用字符串标签"""
        if not top_genes_dict:
            return

        print("    → Groupwise Gene Expression Significance...")

        first_model = list(top_genes_dict.keys())[0]
        top_indices = top_genes_dict[first_model][:15]
        selected_genes = [gene_names[i] for i in top_indices]

        n_classes = len(label_encoder.classes_)
        pval_matrix = np.zeros((len(selected_genes), n_classes * (n_classes - 1) // 2))
        comparison_labels = []

        col_idx = 0
        for i in range(n_classes):
            for j in range(i+1, n_classes):
                class_i = label_encoder.classes_[i]
                class_j = label_encoder.classes_[j]
                comparison_labels.append(f'{class_i}\nvs\n{class_j}')

                # 🔧 修复：直接用字符串标签比较
                mask_i = (y == class_i)
                mask_j = (y == class_j)

                for gene_idx, gene_pos in enumerate(top_indices):
                    data_i = X_scaled[mask_i, gene_pos]
                    data_j = X_scaled[mask_j, gene_pos]

                    if len(data_i) > 1 and len(data_j) > 1:
                        _, pval = stats.mannwhitneyu(data_i, data_j, alternative='two-sided')
                        pval_matrix[gene_idx, col_idx] = pval
                    else:
                        pval_matrix[gene_idx, col_idx] = 1.0

                col_idx += 1

        pval_matrix_log = -np.log10(pval_matrix + 1e-300)
        pval_matrix_log = np.clip(pval_matrix_log, 0, 10)

        fig, ax = plt.subplots(figsize=(max(12, len(comparison_labels)*0.8), 10))

        sns.heatmap(pval_matrix_log, annot=False, cmap='YlOrRd',
                   xticklabels=comparison_labels, yticklabels=selected_genes,
                   cbar_kws={'label': '-log10(p-value)'}, ax=ax, vmin=0, vmax=10)

        for i in range(pval_matrix.shape[0]):
            for j in range(pval_matrix.shape[1]):
                if pval_matrix[i, j] < 0.001:
                    ax.text(j+0.5, i+0.5, '***', ha='center', va='center',
                           color='white', fontsize=12, fontweight='bold')
                elif pval_matrix[i, j] < 0.01:
                    ax.text(j+0.5, i+0.5, '**', ha='center', va='center',
                           color='white', fontsize=10, fontweight='bold')
                elif pval_matrix[i, j] < 0.05:
                    ax.text(j+0.5, i+0.5, '*', ha='center', va='center',
                           color='black', fontsize=8, fontweight='bold')

        ax.set_xlabel('Class Comparisons', fontsize=13, fontweight='bold')
        ax.set_ylabel('Top Genes', fontsize=13, fontweight='bold')
        ax.set_title(f'{trait} - Groupwise Gene Expression Significance\n'
                    f'(Mann-Whitney U test: * p<0.05, ** p<0.01, *** p<0.001)',
                    fontsize=15, fontweight='bold', pad=15)

        plt.xticks(rotation=0, ha='center')
        plt.yticks(rotation=0)
        plt.tight_layout()

        for fmt in ['png', 'pdf']:
            save_path = self.enhanced_dir / f'{trait}_groupwise_significance.{fmt}'
            fig.savefig(save_path, dpi=300 if fmt == 'png' else None, bbox_inches='tight')

        print(f"      ✓ Groupwise Significance Analysis")
        plt.close(fig)

    def _plot_top_gene_importance_boxplots(self, X_scaled: np.ndarray, y: np.ndarray,
                                           gene_names: List[str], top_genes_dict: Dict,
                                           trait: str, label_encoder):
        """每个模型Top10基因的表达箱线图 + 显著性 - 修复版：直接使用字符串标签"""
        if not top_genes_dict:
            return

        print("    → Top Gene Importance Boxplots...")

        for model_name, top_indices in top_genes_dict.items():
            top_10_indices = top_indices[:10]
            top_10_genes = [gene_names[i] for i in top_10_indices]

            n_cols = 5
            n_rows = 2
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            axes = axes.flatten()

            for idx, (gene_idx, gene_name) in enumerate(zip(top_10_indices, top_10_genes)):
                ax = axes[idx]

                gene_expr = X_scaled[:, gene_idx]
                data_for_plot = []
                labels_for_plot = []

                # 🔧 修复：直接使用字符串标签比较
                for class_label in label_encoder.classes_:
                    mask = (y == class_label)  # 直接用字符串比较
                    data_for_plot.append(gene_expr[mask])
                    labels_for_plot.append(class_label)

                valid_indices = [i for i, d in enumerate(data_for_plot) if len(d) > 0]

                if len(valid_indices) < 2:
                    ax.text(0.5, 0.5, f'{gene_name}\nInsufficient data',
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
                    ax.axis('off')
                    continue

                valid_data = [data_for_plot[i] for i in valid_indices]
                valid_labels = [labels_for_plot[i] for i in valid_indices]

                bp = ax.boxplot(valid_data, labels=valid_labels, patch_artist=True,
                               showmeans=True, meanprops=dict(marker='D', markerfacecolor='red',
                                                             markeredgecolor='red', markersize=6))

                colors = plt.cm.Pastel1(np.linspace(0, 1, len(valid_labels)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.8)

                if len(valid_data) > 2:
                    stat, pval = stats.kruskal(*valid_data)

                    if pval < 0.001:
                        sig_text = 'p<0.001***'
                    elif pval < 0.01:
                        sig_text = f'p={pval:.3f}**'
                    elif pval < 0.05:
                        sig_text = f'p={pval:.3f}*'
                    else:
                        sig_text = f'p={pval:.3f}ns'

                    ax.text(0.5, 0.95, sig_text, transform=ax.transAxes,
                           ha='center', va='top', fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

                ax.set_ylabel('Expression (Scaled)', fontsize=10, fontweight='bold')
                ax.set_title(f'#{idx+1}: {gene_name}', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                ax.tick_params(axis='x', rotation=45, labelsize=9)

            plt.suptitle(f'{trait} - {model_name}\nTop 10 Important Genes\n'
                        f'(Kruskal-Wallis test: * p<0.05, ** p<0.01, *** p<0.001)',
                        fontsize=16, fontweight='bold')
            plt.tight_layout()

            for fmt in ['png', 'pdf']:
                save_path = self.enhanced_dir / f'{trait}_{model_name}_top10_genes.{fmt}'
                fig.savefig(save_path, dpi=300 if fmt == 'png' else None, bbox_inches='tight')

            print(f"      ✓ {model_name} Top10 Genes")
            plt.close(fig)

    def _plot_dimensionality_reduction(self, X: np.ndarray, y: np.ndarray,
                                       trait: str, label_encoder,
                                       label_mapping: Optional[Dict] = None):
        """PCA + t-SNE降维 - 修复版：直接使用字符串标签"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)

        ax = axes[0]
        # 🔧 修复：直接用字符串标签比较
        for label in label_encoder.classes_:
            mask = (y == label)
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, alpha=0.6, s=50)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                     fontsize=12, fontweight='bold')
        ax.set_title('PCA Projection', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if len(X) <= 1000:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
            X_tsne = tsne.fit_transform(X)

            ax = axes[1]
            for label in label_encoder.classes_:
                mask = (y == label)
                ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=label, alpha=0.6, s=50)
            ax.set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
            ax.set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
            ax.set_title('t-SNE Projection', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'Dataset too large\nfor t-SNE',
                        ha='center', va='center', fontsize=14)
            axes[1].axis('off')

        plt.suptitle(f'{trait} - Dimensionality Reduction', fontsize=16, fontweight='bold')
        plt.tight_layout()

        for fmt in ['png', 'pdf']:
            save_path = self.enhanced_dir / f'{trait}_dimensionality_reduction.{fmt}'
            fig.savefig(save_path, dpi=300 if fmt == 'png' else None, bbox_inches='tight')

        print(f"    → Dimensionality Reduction")
        plt.close(fig)

    def _plot_true_vs_predicted_scatter(self, dl_results: Dict, X_scaled: np.ndarray,
                                        y: np.ndarray, trait: str, label_encoder):
        """真实vs预测散点图 - 使用数值标签进行预测比较"""
        print("    → True vs Predicted Scatter...")

        valid_models = {k: v for k, v in dl_results.items()
                       if 'predictions' in v and 'true_labels' in v}

        if not valid_models:
            print(f"    ⚠️  无预测数据")
            return

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        n_models = len(valid_models)
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6))

        if n_models == 1:
            axes = [axes]

        for idx, (model_name, results) in enumerate(valid_models.items()):
            ax = axes[idx]

            y_true = results['true_labels']  # 数值标签
            y_pred = results['predictions']  # 数值标签

            n_test = len(y_true)
            X_test_pca = X_pca[-n_test:]

            colors = plt.cm.tab10(np.linspace(0, 1, len(label_encoder.classes_)))

            # 绘制真实值散点
            for class_idx, class_label in enumerate(label_encoder.classes_):
                mask = (y_true == class_idx)
                if np.any(mask):
                    ax.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1],
                              c=[colors[class_idx]], label=f'True: {class_label}',
                              alpha=0.6, s=100, edgecolors='black', linewidths=1.5)

            # 绘制预测中心点
            for class_idx, class_label in enumerate(label_encoder.classes_):
                mask = (y_pred == class_idx)
                if np.any(mask):
                    center = X_test_pca[mask].mean(axis=0)
                    ax.scatter(center[0], center[1], marker='*', s=500,
                              color=colors[class_idx], edgecolors='red',
                              linewidths=3, label=f'Pred Center: {class_label}', zorder=10)

            # 标记错误分类
            errors = (y_true != y_pred)
            if np.any(errors):
                ax.scatter(X_test_pca[errors, 0], X_test_pca[errors, 1],
                          marker='x', s=200, c='red', linewidths=3,
                          label='Misclassified', zorder=11)

            accuracy = self._get_metric(results, 'accuracy', 0.0)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                         fontsize=11, fontweight='bold')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                         fontsize=11, fontweight='bold')
            ax.set_title(f'{model_name}\nTest Acc: {accuracy:.3f}',
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'{trait} - True vs Predicted (Circles=True, Stars=Pred Centers)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        for fmt in ['png', 'pdf']:
            save_path = self.enhanced_dir / f'{trait}_true_vs_predicted.{fmt}'
            fig.savefig(save_path, dpi=300 if fmt == 'png' else None, bbox_inches='tight')

        print(f"    ✓ True vs Predicted Scatter")
        plt.close(fig)

    def _plot_neural_network_architectures(self, models_dict: Dict, trait: str,
                                           training_params: Optional[Dict] = None):
        """神经网络结构图"""
        print("    → Neural Network Architectures...")

        import torch.nn as nn

        for model_name, model in models_dict.items():
            try:
                fig = plt.figure(figsize=(10, 14))
                ax = fig.add_subplot(111)
                ax.axis('off')
                ax.set_xlim(0, 10)
                ax.set_ylim(0, 10)

                layer_info = []
                total_params = 0

                if hasattr(model, 'modules'):
                    for i, layer in enumerate(model.modules()):
                        if isinstance(layer, (nn.Linear, nn.BatchNorm1d, nn.Dropout, nn.ReLU, nn.Softmax)):
                            layer_type = layer.__class__.__name__
                            params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
                            total_params += params

                            info = ""
                            if isinstance(layer, nn.Linear):
                                info = f"{layer.in_features} → {layer.out_features}"
                            elif isinstance(layer, nn.BatchNorm1d):
                                info = f"{layer.num_features} features"
                            elif isinstance(layer, nn.Dropout):
                                info = f"p={layer.p}"

                            layer_info.append({
                                'type': layer_type,
                                'info': info,
                                'params': params
                            })

                if not layer_info:
                    if isinstance(model, nn.Sequential):
                        for i, layer in enumerate(model):
                            layer_type = layer.__class__.__name__
                            params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
                            total_params += params

                            info = ""
                            if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                                info = f"{layer.in_features} → {layer.out_features}"
                            elif hasattr(layer, 'num_features'):
                                info = f"{layer.num_features} features"
                            elif hasattr(layer, 'p'):
                                info = f"p={layer.p}"

                            layer_info.append({
                                'type': layer_type,
                                'info': info,
                                'params': params
                            })

                if not layer_info:
                    title_text = f'{model_name}\n\nNo layer information available'
                    if training_params:
                        title_text += f'\n(Epochs={training_params.get("epochs", "N/A")}, '
                        title_text += f'Dropout={training_params.get("dropout", "N/A")})'

                    ax.text(5, 5, title_text,
                           ha='center', va='center', fontsize=14, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
                    print(f"      ⚠️  {model_name}: 无层信息")
                else:
                    n_layers = len(layer_info)
                    y_positions = np.linspace(8.5, 1.5, n_layers)
                    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_layers))

                    for idx, (layer, y_pos) in enumerate(zip(layer_info, y_positions)):
                        width = 6
                        height = 6.0 / n_layers * 0.7

                        rect = plt.Rectangle((2, y_pos - height/2), width, height,
                                            facecolor=colors[idx], edgecolor='black',
                                            linewidth=2.5, alpha=0.75)
                        ax.add_patch(rect)

                        ax.text(5, y_pos + height/3, layer['type'],
                               ha='center', va='center', fontsize=12,
                               fontweight='bold', color='white')

                        if layer['info']:
                            ax.text(5, y_pos, layer['info'],
                                   ha='center', va='center', fontsize=10,
                                   color='white', style='italic')

                        if layer['params'] > 0:
                            ax.text(5, y_pos - height/3, f"Params: {layer['params']:,}",
                                   ha='center', va='center', fontsize=9,
                                   color='yellow', fontweight='bold')

                        if idx < n_layers - 1:
                            arrow_start_y = y_pos - height/2
                            arrow_end_y = y_positions[idx+1] + height/2
                            ax.annotate('', xy=(5, arrow_end_y), xytext=(5, arrow_start_y),
                                       arrowprops=dict(arrowstyle='->', lw=3, color='black', alpha=0.6))

                    title = f'{model_name} Architecture'
                    ax.text(5, 9.5, title, ha='center', va='top',
                           fontsize=16, fontweight='bold')

                    subtitle = f'Total Parameters: {total_params:,}'
                    if training_params:
                        subtitle += f'\nTrained: {training_params.get("epochs", "N/A")} epochs, '
                        subtitle += f'Dropout={training_params.get("dropout", "N/A")}'

                    ax.text(5, 9.1, subtitle, ha='center', va='top',
                           fontsize=11, color='darkred', fontweight='bold')

                    print(f"      ✓ {model_name} Architecture ({n_layers} layers, {total_params:,} params)")

                plt.tight_layout()

                for fmt in ['png', 'pdf']:
                    save_path = self.enhanced_dir / f'{trait}_{model_name}_architecture.{fmt}'
                    fig.savefig(save_path, dpi=300 if fmt == 'png' else None, bbox_inches='tight')

                plt.close(fig)

            except Exception as e:
                print(f"      ⚠️  {model_name} Architecture failed: {e}")
                import traceback
                traceback.print_exc()
                continue
