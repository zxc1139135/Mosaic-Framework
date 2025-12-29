"""
Data Module
===========

数据加载、预处理和领域聚类模块。

Components:
    - DataLoader: 数据加载器
    - DomainClustering: 领域聚类和专家分配
"""

from .data_loader import (
    DataConfig,
    TextDataset,
    MembershipDataset,
    DomainDataset,
    DataLoaderFactory,
    HuggingFaceDataLoader,
    TextGranularityProcessor,
    AttackDatasetBuilder,
    collate_fn,
)

from .domain_clustering import (
    ClusteringConfig,
    DomainComplexityAnalyzer,
    EmbeddingExtractor,
    SimpleEmbeddingExtractor,
    DomainClusterer,
    DomainManager,
)

__all__ = [
    # Data Loader
    "DataConfig",
    "TextDataset",
    "MembershipDataset",
    "DomainDataset",
    "DataLoaderFactory",
    "HuggingFaceDataLoader",
    "TextGranularityProcessor",
    "AttackDatasetBuilder",
    "collate_fn",
    
    # Domain Clustering
    "ClusteringConfig",
    "DomainComplexityAnalyzer",
    "EmbeddingExtractor",
    "SimpleEmbeddingExtractor",
    "DomainClusterer",
    "DomainManager",
]
