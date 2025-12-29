"""
Domain Clustering Module
========================

领域聚类和专家分配模块，实现论文阶段II中的数据领域划分。

论文参考:
    - 阶段II: 领域专业化训练 (Section 3.1)
    - 基于BERT embeddings的K-means聚类分析
    - 领域复杂度评估 (公式1)
"""

from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter


@dataclass
class ClusteringConfig:
    """聚类配置"""
    num_clusters: int = 8  # K* = 8
    random_state: int = 42
    max_iter: int = 300
    n_init: int = 10
    batch_size: int = 1024
    use_pca: bool = True
    pca_components: int = 128


class DomainComplexityAnalyzer:
    """
    领域复杂度分析器
    
    论文公式(1): H_k = α1 * (1 - TTR_k/max_j TTR_j) + 
                        α2 * (Var(SentLength_k) + Var(TreeDepth_k))/2 +
                        α3 * (-Σ p_k,t log p_k,t)
    """
    
    def __init__(
        self,
        alpha1: float = 0.4,  # 词汇多样性权重
        alpha2: float = 0.3,  # 句法复杂度权重
        alpha3: float = 0.3,  # 语义熵权重
    ):
        """
        Args:
            alpha1: 词汇多样性权重
            alpha2: 句法复杂度权重
            alpha3: 语义熵权重
        """
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        
    def compute_type_token_ratio(self, texts: List[str]) -> float:
        """
        计算Type-Token Ratio (TTR)
        
        TTR = 不同词的数量 / 总词数
        
        Args:
            texts: 文本列表
            
        Returns:
            ttr: Type-Token Ratio
        """
        all_tokens = []
        for text in texts:
            tokens = text.lower().split()
            all_tokens.extend(tokens)
            
        if len(all_tokens) == 0:
            return 0.0
            
        unique_tokens = set(all_tokens)
        return len(unique_tokens) / len(all_tokens)
    
    def compute_sentence_length_variance(self, texts: List[str]) -> float:
        """
        计算句子长度方差
        
        Args:
            texts: 文本列表
            
        Returns:
            variance: 句子长度方差
        """
        all_lengths = []
        
        for text in texts:
            # 简单句子分割
            sentences = text.replace("!", ".").replace("?", ".").split(".")
            for sent in sentences:
                words = sent.strip().split()
                if len(words) > 0:
                    all_lengths.append(len(words))
                    
        if len(all_lengths) == 0:
            return 0.0
            
        return float(np.var(all_lengths))
    
    def compute_tree_depth_variance(self, texts: List[str]) -> float:
        """
        计算句法树深度方差 (简化实现)
        
        使用嵌套标点符号深度作为近似
        
        Args:
            texts: 文本列表
            
        Returns:
            variance: 树深度方差
        """
        depths = []
        
        for text in texts:
            # 使用括号和逗号嵌套作为深度近似
            max_depth = 0
            current_depth = 0
            
            for char in text:
                if char in "([{":
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
                elif char in ")]}":
                    current_depth = max(0, current_depth - 1)
                elif char == ",":
                    current_depth += 0.5
                    
            depths.append(max_depth)
            
        if len(depths) == 0:
            return 0.0
            
        return float(np.var(depths))
    
    def compute_semantic_entropy(
        self,
        texts: List[str],
        num_topics: int = 10,
    ) -> float:
        """
        计算语义熵 (基于简单词频分布)
        
        Args:
            texts: 文本列表
            num_topics: 主题数量
            
        Returns:
            entropy: 语义熵
        """
        # 使用简化的词频分布作为语义近似
        word_counts = Counter()
        
        for text in texts:
            tokens = text.lower().split()
            word_counts.update(tokens)
            
        if len(word_counts) == 0:
            return 0.0
            
        # 计算词频分布的熵
        total = sum(word_counts.values())
        probs = [count / total for count in word_counts.values()]
        
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        
        # 归一化
        max_entropy = np.log(len(word_counts))
        if max_entropy > 0:
            entropy = entropy / max_entropy
            
        return entropy
    
    def compute_complexity(self, texts: List[str]) -> float:
        """
        计算领域复杂度
        
        Args:
            texts: 领域文本列表
            
        Returns:
            complexity: 综合复杂度分数
        """
        # 1. 词汇多样性 (TTR越低，复杂度越高)
        ttr = self.compute_type_token_ratio(texts)
        vocab_complexity = 1 - ttr  # 反转，因为低TTR表示高重复性
        
        # 2. 句法复杂度
        sent_var = self.compute_sentence_length_variance(texts)
        tree_var = self.compute_tree_depth_variance(texts)
        syntax_complexity = (sent_var + tree_var) / 2
        
        # 归一化句法复杂度
        syntax_complexity = min(syntax_complexity / 100, 1.0)
        
        # 3. 语义熵
        semantic_entropy = self.compute_semantic_entropy(texts)
        
        # 加权组合
        complexity = (
            self.alpha1 * vocab_complexity +
            self.alpha2 * syntax_complexity +
            self.alpha3 * semantic_entropy
        )
        
        return complexity


class EmbeddingExtractor:
    """
    文本嵌入提取器
    
    使用预训练模型提取文本的向量表示
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
    ):
        """
        Args:
            model_name: 预训练模型名称
            device: 计算设备
            batch_size: 批次大小
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        
    def _load_model(self):
        """延迟加载模型"""
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model = self._model.to(self.device)
            self._model.eval()
            
    @property
    def model(self):
        self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        self._load_model()
        return self._tokenizer
    
    @torch.no_grad()
    def extract_embeddings(
        self,
        texts: List[str],
        max_length: int = 512,
        pooling: str = "mean",
    ) -> np.ndarray:
        """
        提取文本嵌入
        
        Args:
            texts: 文本列表
            max_length: 最大序列长度
            pooling: 池化方式
            
        Returns:
            embeddings: 嵌入向量 [N, hidden_dim]
        """
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            # 前向传播
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            
            # 池化
            if pooling == "mean":
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                embeddings = sum_hidden / sum_mask
            elif pooling == "cls":
                embeddings = hidden_states[:, 0, :]
            else:
                raise ValueError(f"Unknown pooling: {pooling}")
                
            all_embeddings.append(embeddings.cpu().numpy())
            
        return np.vstack(all_embeddings)


class SimpleEmbeddingExtractor:
    """
    简化嵌入提取器 (不依赖transformer)
    
    使用TF-IDF或词频向量
    """
    
    def __init__(self, max_features: int = 5000):
        """
        Args:
            max_features: 最大特征数
        """
        self.max_features = max_features
        self._vectorizer = None
        
    def _build_vocabulary(self, texts: List[str]):
        """构建词汇表"""
        word_counts = Counter()
        
        for text in texts:
            tokens = text.lower().split()
            word_counts.update(tokens)
            
        # 选择最常见的词
        most_common = word_counts.most_common(self.max_features)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(most_common)}
        
    def extract_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        提取TF-IDF风格的嵌入
        
        Args:
            texts: 文本列表
            
        Returns:
            embeddings: 嵌入向量 [N, max_features]
        """
        if not hasattr(self, 'vocabulary'):
            self._build_vocabulary(texts)
            
        embeddings = np.zeros((len(texts), len(self.vocabulary)))
        
        # 计算文档频率
        doc_freq = Counter()
        for text in texts:
            unique_tokens = set(text.lower().split())
            for token in unique_tokens:
                if token in self.vocabulary:
                    doc_freq[token] += 1
                    
        # 计算IDF
        num_docs = len(texts)
        idf = {
            word: np.log(num_docs / (freq + 1))
            for word, freq in doc_freq.items()
        }
        
        # 计算TF-IDF
        for i, text in enumerate(texts):
            tokens = text.lower().split()
            tf = Counter(tokens)
            
            for token, count in tf.items():
                if token in self.vocabulary:
                    idx = self.vocabulary[token]
                    tf_value = count / len(tokens)
                    idf_value = idf.get(token, 0)
                    embeddings[i, idx] = tf_value * idf_value
                    
        return embeddings


class DomainClusterer:
    """
    领域聚类器
    
    将训练数据划分为K个语义相关的领域子集
    """
    
    def __init__(self, config: ClusteringConfig):
        """
        Args:
            config: 聚类配置
        """
        self.config = config
        self.kmeans = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=config.pca_components) if config.use_pca else None
        self.cluster_centers_ = None
        self.labels_ = None
        
    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """
        拟合聚类模型
        
        Args:
            embeddings: 文本嵌入 [N, hidden_dim]
            
        Returns:
            labels: 聚类标签 [N]
        """
        # 标准化
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # PCA降维
        if self.pca is not None:
            n_components = min(self.config.pca_components, embeddings_scaled.shape[1])
            self.pca = PCA(n_components=n_components)
            embeddings_reduced = self.pca.fit_transform(embeddings_scaled)
        else:
            embeddings_reduced = embeddings_scaled
            
        # K-Means聚类
        if len(embeddings_reduced) > 10000:
            # 大数据集使用Mini-Batch K-Means
            self.kmeans = MiniBatchKMeans(
                n_clusters=self.config.num_clusters,
                random_state=self.config.random_state,
                batch_size=self.config.batch_size,
                max_iter=self.config.max_iter,
            )
        else:
            self.kmeans = KMeans(
                n_clusters=self.config.num_clusters,
                random_state=self.config.random_state,
                n_init=self.config.n_init,
                max_iter=self.config.max_iter,
            )
            
        self.labels_ = self.kmeans.fit_predict(embeddings_reduced)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        
        return self.labels_
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        预测新样本的领域
        
        Args:
            embeddings: 文本嵌入
            
        Returns:
            labels: 聚类标签
        """
        embeddings_scaled = self.scaler.transform(embeddings)
        
        if self.pca is not None:
            embeddings_reduced = self.pca.transform(embeddings_scaled)
        else:
            embeddings_reduced = embeddings_scaled
            
        return self.kmeans.predict(embeddings_reduced)
    
    def get_cluster_sizes(self) -> Dict[int, int]:
        """获取每个聚类的样本数量"""
        if self.labels_ is None:
            return {}
        return dict(Counter(self.labels_))
    
    def get_cluster_proportions(self) -> Dict[int, float]:
        """获取每个聚类的样本比例"""
        sizes = self.get_cluster_sizes()
        total = sum(sizes.values())
        return {k: v / total for k, v in sizes.items()}


class DomainManager:
    """
    领域管理器
    
    整合聚类、复杂度分析和专家分配
    """
    
    def __init__(
        self,
        num_domains: int = 8,
        use_bert: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            num_domains: 领域数量
            use_bert: 是否使用BERT嵌入
            device: 计算设备
        """
        self.num_domains = num_domains
        self.device = device
        
        # 嵌入提取器
        if use_bert:
            self.embedding_extractor = EmbeddingExtractor(device=device)
        else:
            self.embedding_extractor = SimpleEmbeddingExtractor()
            
        # 聚类器
        self.clustering_config = ClusteringConfig(num_clusters=num_domains)
        self.clusterer = DomainClusterer(self.clustering_config)
        
        # 复杂度分析器
        self.complexity_analyzer = DomainComplexityAnalyzer()
        
        # 领域信息
        self.domain_texts: Dict[int, List[str]] = {}
        self.domain_complexities: Dict[int, float] = {}
        self.domain_proportions: Dict[int, float] = {}
        
    def cluster_texts(self, texts: List[str]) -> Dict[int, List[str]]:
        """
        对文本进行领域聚类
        
        Args:
            texts: 文本列表
            
        Returns:
            domain_texts: {domain_id: [texts]} 字典
        """
        # 提取嵌入
        print("Extracting embeddings...")
        embeddings = self.embedding_extractor.extract_embeddings(texts)
        
        # 聚类
        print("Clustering...")
        labels = self.clusterer.fit(embeddings)
        
        # 按领域组织文本
        self.domain_texts = {i: [] for i in range(self.num_domains)}
        for text, label in zip(texts, labels):
            self.domain_texts[label].append(text)
            
        # 更新比例
        self.domain_proportions = self.clusterer.get_cluster_proportions()
        
        return self.domain_texts
    
    def analyze_domains(self) -> Dict[int, float]:
        """
        分析每个领域的复杂度
        
        Returns:
            complexities: {domain_id: complexity} 字典
        """
        print("Analyzing domain complexities...")
        
        for domain_id, texts in self.domain_texts.items():
            if len(texts) > 0:
                complexity = self.complexity_analyzer.compute_complexity(texts)
                self.domain_complexities[domain_id] = complexity
            else:
                self.domain_complexities[domain_id] = 0.0
                
        return self.domain_complexities
    
    def compute_expert_params(self, total_params: int) -> Dict[int, int]:
        """
        计算每个专家的参数分配
        
        论文公式(2): θ*_k = Θ_total * (p_k * H_k) / Σ(p_j * H_j)
        
        Args:
            total_params: 总参数预算
            
        Returns:
            expert_params: {expert_id: params} 字典
        """
        # 确保已分析复杂度
        if not self.domain_complexities:
            self.analyze_domains()
            
        # 计算加权分数
        weighted_scores = {}
        for domain_id in range(self.num_domains):
            p_k = self.domain_proportions.get(domain_id, 0)
            h_k = self.domain_complexities.get(domain_id, 0)
            weighted_scores[domain_id] = p_k * h_k
            
        # 归一化
        total_score = sum(weighted_scores.values())
        if total_score == 0:
            # 均匀分配
            return {i: total_params // self.num_domains for i in range(self.num_domains)}
            
        expert_params = {
            domain_id: int(total_params * score / total_score)
            for domain_id, score in weighted_scores.items()
        }
        
        return expert_params
    
    def get_domain_summary(self) -> Dict:
        """获取领域摘要信息"""
        summary = {
            "num_domains": self.num_domains,
            "domains": {}
        }
        
        for domain_id in range(self.num_domains):
            summary["domains"][domain_id] = {
                "num_texts": len(self.domain_texts.get(domain_id, [])),
                "proportion": self.domain_proportions.get(domain_id, 0),
                "complexity": self.domain_complexities.get(domain_id, 0),
            }
            
        return summary
    
    def assign_text_to_domain(self, text: str) -> int:
        """
        为新文本分配领域
        
        Args:
            text: 输入文本
            
        Returns:
            domain_id: 领域ID
        """
        embeddings = self.embedding_extractor.extract_embeddings([text])
        labels = self.clusterer.predict(embeddings)
        return int(labels[0])


if __name__ == "__main__":
    # 测试代码
    print("Testing Domain Clustering Module...")
    
    # 创建测试数据
    texts = [
        # 科技领域
        "Machine learning algorithms can analyze large datasets efficiently.",
        "Deep neural networks have achieved state-of-the-art results.",
        "Artificial intelligence is transforming many industries.",
        "Computer vision systems can recognize objects in images.",
        
        # 医学领域
        "The patient was diagnosed with a rare genetic disorder.",
        "Clinical trials showed promising results for the new drug.",
        "Medical imaging technology has improved disease detection.",
        "Healthcare systems are adopting electronic health records.",
        
        # 文学领域
        "The novel explores themes of love and loss.",
        "Poetry allows for creative expression of emotions.",
        "The author's writing style is both elegant and accessible.",
        "Literary analysis reveals hidden meanings in the text.",
        
        # 商业领域
        "The company reported strong quarterly earnings.",
        "Market analysis suggests growing demand for the product.",
        "Strategic investments have led to significant returns.",
        "Consumer behavior patterns are changing rapidly.",
    ] * 50
    
    # 测试DomainComplexityAnalyzer
    print("Testing DomainComplexityAnalyzer...")
    analyzer = DomainComplexityAnalyzer()
    
    ttr = analyzer.compute_type_token_ratio(texts[:4])
    print(f"TTR: {ttr:.4f}")
    
    sent_var = analyzer.compute_sentence_length_variance(texts[:4])
    print(f"Sentence length variance: {sent_var:.4f}")
    
    complexity = analyzer.compute_complexity(texts[:4])
    print(f"Domain complexity: {complexity:.4f}")
    
    # 测试SimpleEmbeddingExtractor
    print("\nTesting SimpleEmbeddingExtractor...")
    simple_extractor = SimpleEmbeddingExtractor(max_features=100)
    simple_embeddings = simple_extractor.extract_embeddings(texts)
    print(f"Simple embeddings shape: {simple_embeddings.shape}")
    
    # 测试DomainClusterer
    print("\nTesting DomainClusterer...")
    config = ClusteringConfig(num_clusters=4, use_pca=True, pca_components=50)
    clusterer = DomainClusterer(config)
    labels = clusterer.fit(simple_embeddings)
    print(f"Cluster sizes: {clusterer.get_cluster_sizes()}")
    print(f"Cluster proportions: {clusterer.get_cluster_proportions()}")
    
    # 测试DomainManager (使用简化嵌入)
    print("\nTesting DomainManager...")
    manager = DomainManager(num_domains=4, use_bert=False)
    domain_texts = manager.cluster_texts(texts)
    
    for domain_id, domain_text_list in domain_texts.items():
        print(f"Domain {domain_id}: {len(domain_text_list)} texts")
        
    complexities = manager.analyze_domains()
    print(f"Domain complexities: {complexities}")
    
    expert_params = manager.compute_expert_params(total_params=1_000_000)
    print(f"Expert params: {expert_params}")
    
    summary = manager.get_domain_summary()
    print(f"Domain summary: {summary}")
    
    print("\nAll tests passed!")
