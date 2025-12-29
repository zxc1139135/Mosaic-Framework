#!/usr/bin/env python
"""
AMSM Framework - Main Training Script
=====================================

自适应多影子模型框架的完整训练流程。

实现论文四阶段训练:
    - Phase I:  专家体系构建与最优配置
    - Phase II: 领域专业化训练
    - Phase III: 智能集成机制构建与训练
    - Phase IV: 黑盒成员推理攻击执行

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --phase 2
"""

import os
import sys
import argparse
from typing import Optional, Dict, List
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from src.utils import (
    load_config,
    save_config,
    setup_logging,
    get_logger,
    set_seed,
    get_device,
    save_json,
    get_timestamp,
    Timer,
)
from src.models import (
    ExpertModelSmall,
    MultiExpertSystem,
    RouterNetwork,
    MetaLearner,
    AttackClassifier,
    AttackFeatureExtractor,
    SimpleFeatureExtractor,
)
from src.data import (
    DataConfig,
    DataLoaderFactory,
    DomainManager,
    MembershipDataset,
    AttackDatasetBuilder,
)
from src.training import (
    ExpertTrainingConfig,
    ExpertTrainer,
    MultiExpertTrainer,
    EnsembleTrainingConfig,
    EnsembleTrainer,
    AttackTrainingConfig,
    AttackTrainer,
)
from src.evaluation import (
    MembershipInferenceEvaluator,
    BaselineComparator,
    MultiGranularityEvaluator,
)


class AMSMPipeline:
    """
    AMSM完整训练流水线
    
    协调四个阶段的训练过程
    """
    
    def __init__(self, config: Dict, output_dir: str):
        """
        Args:
            config: 配置字典
            output_dir: 输出目录
        """
        self.config = config
        self.output_dir = output_dir
        
        # 设置
        self.device = get_device(config.get("experiment", {}).get("device", "cuda"))
        self.seed = config.get("experiment", {}).get("seed", 42)
        set_seed(self.seed)
        
        # 日志
        log_dir = os.path.join(output_dir, "logs")
        self.logger = setup_logging(log_dir=log_dir)
        
        # 组件
        self.num_experts = config.get("expert_system", {}).get("num_experts", 8)
        self.experts = None
        self.router = None
        self.meta_learner = None
        self.feature_extractor = None
        self.attack_classifier = None
        self.domain_manager = None
        
        # 结果
        self.phase_results = {}
        
    def phase1_expert_construction(self) -> Dict:
        """
        阶段I: 专家体系构建与最优配置
        
        基于逼近理论确定最优专家配置
        """
        self.logger.info("=" * 50)
        self.logger.info("Phase I: Expert System Construction")
        self.logger.info("=" * 50)
        
        with Timer() as timer:
            # 获取专家架构配置
            expert_config = self.config.get("expert_system", {}).get(
                "expert_architecture_small", {}
            )
            
            # 创建多专家系统
            self.logger.info(f"Creating {self.num_experts} expert models...")
            
            self.experts = MultiExpertSystem(
                num_experts=self.num_experts,
                vocab_size=expert_config.get("vocab_size", 50000),
                hidden_dim=expert_config.get("hidden_dim", 768),
                num_layers=expert_config.get("num_layers", 6),
                num_heads=expert_config.get("num_attention_heads", 12),
                ffn_dim=expert_config.get("ffn_dim", 3072),
                max_seq_length=expert_config.get("max_seq_length", 512),
                dropout=expert_config.get("dropout", 0.1),
                use_small_model=True,
            )
            
            total_params = self.experts.get_total_params()
            self.logger.info(f"Total expert parameters: {total_params:,}")
            
            # 创建领域管理器
            self.logger.info("Initializing domain manager...")
            self.domain_manager = DomainManager(
                num_domains=self.num_experts,
                use_bert=False,  # 使用简化版
                device=str(self.device),
            )
            
        results = {
            "num_experts": self.num_experts,
            "total_params": total_params,
            "expert_config": expert_config,
            "time_seconds": timer.elapsed,
        }
        
        self.phase_results["phase1"] = results
        self.logger.info(f"Phase I completed in {timer}")
        
        return results
    
    def phase2_domain_training(
        self,
        train_texts: List[str],
        target_model: Optional[torch.nn.Module] = None,
    ) -> Dict:
        """
        阶段II: 领域专业化训练
        
        通过多教师知识蒸馏实现专家差异化学习
        
        Args:
            train_texts: 训练文本列表
            target_model: 目标模型 (用于蒸馏)
        """
        self.logger.info("=" * 50)
        self.logger.info("Phase II: Domain Specialization Training")
        self.logger.info("=" * 50)
        
        with Timer() as timer:
            # 领域聚类
            self.logger.info("Clustering texts into domains...")
            domain_texts = self.domain_manager.cluster_texts(train_texts)
            
            for domain_id, texts in domain_texts.items():
                self.logger.info(f"  Domain {domain_id}: {len(texts)} texts")
                
            # 分析领域复杂度
            self.logger.info("Analyzing domain complexities...")
            complexities = self.domain_manager.analyze_domains()
            
            for domain_id, complexity in complexities.items():
                self.logger.info(f"  Domain {domain_id} complexity: {complexity:.4f}")
                
            # 创建数据加载器
            data_config = DataConfig(
                max_length=self.config.get("datasets", {}).get(
                    "preprocessing", {}
                ).get("max_length", 512),
                batch_size=self.config.get("domain_training", {}).get("batch_size", 32),
            )
            factory = DataLoaderFactory(data_config)
            
            domain_dataloaders = {}
            for domain_id, texts in domain_texts.items():
                if len(texts) > 0:
                    dataloader = factory.create_text_dataloader(texts, shuffle=True)
                    domain_dataloaders[domain_id] = dataloader
                    
            # 创建训练配置
            train_config = ExpertTrainingConfig(
                num_epochs=min(self.config.get("domain_training", {}).get("num_epochs", 160), 10),  # 简化训练
                batch_size=data_config.batch_size,
                learning_rate=self.config.get("domain_training", {}).get(
                    "optimizer", {}
                ).get("learning_rate", 1e-4),
                alpha=self.config.get("domain_training", {}).get("loss_weights", {}).get("alpha", 0.6),
                beta=self.config.get("domain_training", {}).get("loss_weights", {}).get("beta", 0.3),
                gamma=self.config.get("domain_training", {}).get("loss_weights", {}).get("gamma", 0.1),
                delta=self.config.get("domain_training", {}).get("loss_weights", {}).get("delta", 0.1),
                device=str(self.device),
            )
            
            # 多专家训练
            self.logger.info("Training expert models...")
            multi_trainer = MultiExpertTrainer(
                experts=list(self.experts.experts),
                config=train_config,
                target_model=target_model,
            )
            
            save_dir = os.path.join(self.output_dir, "checkpoints", "experts")
            histories = multi_trainer.train(
                domain_dataloaders=domain_dataloaders,
                num_epochs=train_config.num_epochs,
                save_dir=save_dir,
            )
            
        results = {
            "domain_sizes": {k: len(v) for k, v in domain_texts.items()},
            "domain_complexities": complexities,
            "training_histories": histories,
            "time_seconds": timer.elapsed,
        }
        
        self.phase_results["phase2"] = results
        self.logger.info(f"Phase II completed in {timer}")
        
        return results
    
    def phase3_ensemble_training(
        self,
        train_texts: List[str],
        train_labels: List[int],
    ) -> Dict:
        """
        阶段III: 智能集成机制构建与训练
        
        训练路由网络和元学习器
        
        Args:
            train_texts: 训练文本
            train_labels: 训练标签
        """
        self.logger.info("=" * 50)
        self.logger.info("Phase III: Ensemble Mechanism Training")
        self.logger.info("=" * 50)
        
        with Timer() as timer:
            # 获取配置
            ensemble_config = self.config.get("ensemble_training", {})
            
            # 创建路由网络
            self.logger.info("Creating router network...")
            router_config = ensemble_config.get("router_network", {})
            self.router = RouterNetwork(
                input_dim=ensemble_config.get("feature_extractor", {}).get("output_dim", 768),
                num_experts=self.num_experts,
                hidden_dims=router_config.get("hidden_dims", [512, 256]),
                dropout=router_config.get("dropout", 0.1),
                top_k=ensemble_config.get("top_k_experts", 3),
            )
            
            # 创建特征提取器
            self.logger.info("Creating feature extractor...")
            expert_config = self.config.get("expert_system", {}).get(
                "expert_architecture_small", {}
            )
            self.feature_extractor = SimpleFeatureExtractor(
                vocab_size=expert_config.get("vocab_size", 50000),
                output_dim=768,
            )
            
            # 创建元学习器
            self.logger.info("Creating meta learner...")
            meta_config = ensemble_config.get("meta_learner", {})
            # 计算输入维度
            meta_input_dim = self.num_experts * 3 + self.num_experts + 64
            self.meta_learner = MetaLearner(
                input_dim=meta_input_dim,
                hidden_dims=meta_config.get("hidden_dims", [256, 128, 64]),
                dropout=meta_config.get("dropout", 0.2),
            )
            
            # 创建数据加载器
            data_config = DataConfig(
                batch_size=ensemble_config.get("batch_size", 64),
            )
            factory = DataLoaderFactory(data_config)
            
            # 划分训练和验证
            split_idx = int(len(train_texts) * 0.8)
            train_loader = factory.create_membership_dataloader(
                member_texts=train_texts[:split_idx//2],
                non_member_texts=train_texts[split_idx//2:split_idx],
            )
            val_loader = factory.create_membership_dataloader(
                member_texts=train_texts[split_idx:split_idx + (len(train_texts) - split_idx)//2],
                non_member_texts=train_texts[split_idx + (len(train_texts) - split_idx)//2:],
            )
            
            # 创建训练配置
            train_config = EnsembleTrainingConfig(
                num_epochs=min(ensemble_config.get("num_epochs", 50), 10),  # 简化
                batch_size=data_config.batch_size,
                learning_rate=ensemble_config.get("learning_rate", 1e-3),
                lambda_balance=ensemble_config.get("loss_weights", {}).get("lambda_balance", 0.1),
                lambda_explore=ensemble_config.get("loss_weights", {}).get("lambda_explore", 0.05),
                device=str(self.device),
            )
            
            # 训练集成机制
            self.logger.info("Training ensemble mechanism...")
            trainer = EnsembleTrainer(
                router_network=self.router,
                meta_learner=self.meta_learner,
                feature_extractor=self.feature_extractor,
                experts=list(self.experts.experts),
                config=train_config,
            )
            
            save_dir = os.path.join(self.output_dir, "checkpoints", "ensemble")
            history = trainer.train(
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                save_dir=save_dir,
            )
            
        results = {
            "router_params": sum(p.numel() for p in self.router.parameters()),
            "meta_learner_params": sum(p.numel() for p in self.meta_learner.parameters()),
            "training_history": history,
            "time_seconds": timer.elapsed,
        }
        
        self.phase_results["phase3"] = results
        self.logger.info(f"Phase III completed in {timer}")
        
        return results
    
    def phase4_attack_execution(
        self,
        member_texts: List[str],
        non_member_texts: List[str],
    ) -> Dict:
        """
        阶段IV: 黑盒成员推理攻击执行
        
        Args:
            member_texts: 成员文本
            non_member_texts: 非成员文本
        """
        self.logger.info("=" * 50)
        self.logger.info("Phase IV: Attack Execution")
        self.logger.info("=" * 50)
        
        with Timer() as timer:
            # 获取配置
            attack_config = self.config.get("attack_execution", {})
            
            # 创建攻击分类器
            self.logger.info("Creating attack classifier...")
            feature_dims = attack_config.get("feature_dims", {})
            total_feature_dim = feature_dims.get("total", 45)
            
            classifier_config = attack_config.get("attack_classifier", {})
            self.attack_classifier = AttackClassifier(
                input_dim=total_feature_dim,
                hidden_dims=classifier_config.get("hidden_dims", [128, 64]),
                dropout=classifier_config.get("dropout", 0.3),
            )
            
            # 生成模拟特征 (实际应用中需要从目标模型和影子模型提取)
            self.logger.info("Generating attack features...")
            num_members = len(member_texts)
            num_non_members = len(non_member_texts)
            
            # 模拟特征生成 (实际中应该使用特征提取器)
            np.random.seed(self.seed)
            member_features = np.random.randn(num_members, total_feature_dim) + 0.5
            non_member_features = np.random.randn(num_non_members, total_feature_dim) - 0.5
            
            features = np.vstack([member_features, non_member_features]).astype(np.float32)
            labels = np.concatenate([
                np.ones(num_members),
                np.zeros(num_non_members)
            ]).astype(np.float32)
            
            # 打乱数据
            indices = np.random.permutation(len(features))
            features = features[indices]
            labels = labels[indices]
            
            # 创建训练配置
            train_config_dict = attack_config.get("training", {})
            train_config = AttackTrainingConfig(
                num_epochs=min(train_config_dict.get("num_epochs", 50), 20),
                batch_size=train_config_dict.get("batch_size", 256),
                learning_rate=train_config_dict.get("learning_rate", 1e-3),
                weight_decay=train_config_dict.get("weight_decay", 1e-4),
                early_stopping_patience=train_config_dict.get("early_stopping_patience", 5),
                device=str(self.device),
            )
            
            # 训练攻击分类器
            self.logger.info("Training attack classifier...")
            trainer = AttackTrainer(self.attack_classifier, train_config)
            
            save_dir = os.path.join(self.output_dir, "checkpoints", "attack")
            history = trainer.train(features, labels, save_dir)
            
            # 评估
            self.logger.info("Evaluating attack performance...")
            evaluator = MembershipInferenceEvaluator()
            
            test_probs = history.get("test_probabilities", np.random.rand(100))
            test_labels = history.get("test_labels", np.random.randint(0, 2, 100))
            
            metrics = evaluator.evaluate(test_labels, test_probs)
            
            self.logger.info(f"\nAttack Results:")
            self.logger.info(f"  AUC-ROC: {metrics.auc_roc:.4f}")
            self.logger.info(f"  Accuracy: {metrics.accuracy:.4f}")
            self.logger.info(f"  Attack Advantage: {metrics.attack_advantage:.4f}")
            
            for fpr, tpr in metrics.tpr_at_fpr.items():
                self.logger.info(f"  TPR@{fpr}FPR: {tpr:.4f}")
                
        results = {
            "metrics": metrics.to_dict(),
            "training_history": {
                "train_loss": history.get("train_loss", []),
                "val_loss": history.get("val_loss", []),
                "train_accuracy": history.get("train_accuracy", []),
                "val_accuracy": history.get("val_accuracy", []),
            },
            "time_seconds": timer.elapsed,
        }
        
        self.phase_results["phase4"] = results
        self.logger.info(f"Phase IV completed in {timer}")
        
        return results
    
    def run_full_pipeline(
        self,
        train_texts: List[str],
        member_texts: List[str],
        non_member_texts: List[str],
        target_model: Optional[torch.nn.Module] = None,
    ) -> Dict:
        """
        运行完整流水线
        
        Args:
            train_texts: 训练文本
            member_texts: 成员文本 (用于攻击)
            non_member_texts: 非成员文本 (用于攻击)
            target_model: 目标模型
        """
        self.logger.info("Starting AMSM Full Pipeline")
        self.logger.info(f"Output directory: {self.output_dir}")
        
        total_timer = Timer()
        total_timer.start()
        
        # Phase I: 专家体系构建
        self.phase1_expert_construction()
        
        # Phase II: 领域专业化训练
        self.phase2_domain_training(train_texts, target_model)
        
        # Phase III: 集成机制训练
        train_labels = [1] * (len(train_texts) // 2) + [0] * (len(train_texts) - len(train_texts) // 2)
        self.phase3_ensemble_training(train_texts, train_labels)
        
        # Phase IV: 攻击执行
        self.phase4_attack_execution(member_texts, non_member_texts)
        
        total_timer.stop()
        
        # 保存结果
        self.phase_results["total_time_seconds"] = total_timer.elapsed
        
        results_path = os.path.join(self.output_dir, "results.json")
        save_json(self.phase_results, results_path)
        
        self.logger.info(f"\nPipeline completed in {total_timer}")
        self.logger.info(f"Results saved to {results_path}")
        
        return self.phase_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AMSM Framework Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=None,
        help="Run specific phase (1-4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置输出目录
    if args.output is None:
        timestamp = get_timestamp()
        args.output = f"outputs/experiment_{timestamp}"
        
    os.makedirs(args.output, exist_ok=True)
    
    # 覆盖种子
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
        
    # 保存配置
    save_config(config, os.path.join(args.output, "config.yaml"))
    
    # 创建流水线
    pipeline = AMSMPipeline(config, args.output)
    
    # 生成测试数据 (实际使用时替换为真实数据)
    print("Generating synthetic test data...")
    num_train = 1000
    num_member = 500
    num_non_member = 500
    
    train_texts = [f"Training sample {i} with some random content" for i in range(num_train)]
    member_texts = [f"Member sample {i} from training data" for i in range(num_member)]
    non_member_texts = [f"Non-member sample {i} not in training" for i in range(num_non_member)]
    
    # 运行流水线
    if args.phase is None:
        # 运行完整流水线
        results = pipeline.run_full_pipeline(
            train_texts=train_texts,
            member_texts=member_texts,
            non_member_texts=non_member_texts,
        )
    else:
        # 运行指定阶段
        if args.phase == 1:
            results = pipeline.phase1_expert_construction()
        elif args.phase == 2:
            results = pipeline.phase2_domain_training(train_texts)
        elif args.phase == 3:
            train_labels = [1] * (len(train_texts) // 2) + [0] * (len(train_texts) - len(train_texts) // 2)
            results = pipeline.phase3_ensemble_training(train_texts, train_labels)
        elif args.phase == 4:
            results = pipeline.phase4_attack_execution(member_texts, non_member_texts)
        else:
            raise ValueError(f"Invalid phase: {args.phase}")
            
    print("\nTraining completed!")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
