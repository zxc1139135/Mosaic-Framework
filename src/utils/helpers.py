"""
Helper Functions

Utility functions for training, evaluation, and experiment management.
"""

import os
import sys
import json
import random
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np
import torch

# ==================== ====================

def load_config(config_path: str) -> Dict[str, Any]:
 """
 YAML
 
 Args:
 config_path: 
 
 Returns:
 config: 
 """
 with open(config_path, 'r', encoding='utf-8') as f:
 config = yaml.safe_load(f)
 return config

def save_config(config: Dict[str, Any], save_path: str):
 """
 YAML
 
 Args:
 config: 
 save_path: 
 """
 os.makedirs(os.path.dirname(save_path), exist_ok=True)
 with open(save_path, 'w', encoding='utf-8') as f:
 yaml.dump(config, f, default_flow_style=False)

def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
 """
 
 
 Args:
 base_config: 
 override_config: 
 
 Returns:
 merged: 
 """
 merged = base_config.copy()
 
 for key, value in override_config.items():
 if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
 merged[key] = merge_configs(merged[key], value)
 else:
 merged[key] = value
 
 return merged

# ==================== ====================

def setup_logging(
 log_dir: Optional[str] = None,
 log_level: str = "INFO",
 log_file: Optional[str] = None,
) -> logging.Logger:
 """
 
 
 Args:
 log_dir: 
 log_level: 
 log_file: 
 
 Returns:
 logger: 
 """
 logger = logging.getLogger("Mosaic")
 logger.setLevel(getattr(logging, log_level.upper()))
 
 # handler
 logger.handlers.clear()
 
 # handler
 console_handler = logging.StreamHandler(sys.stdout)
 console_handler.setLevel(logging.INFO)
 console_format = logging.Formatter(
 '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
 datefmt='%Y-%m-%d %H:%M:%S'
 )
 console_handler.setFormatter(console_format)
 logger.addHandler(console_handler)
 
 # handler
 if log_dir is not None:
 os.makedirs(log_dir, exist_ok=True)
 
 if log_file is None:
 timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
 log_file = f"mosaic_{timestamp}.log"
 
 file_handler = logging.FileHandler(
 os.path.join(log_dir, log_file),
 encoding='utf-8'
 )
 file_handler.setLevel(logging.DEBUG)
 file_format = logging.Formatter(
 '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
 )
 file_handler.setFormatter(file_format)
 logger.addHandler(file_handler)
 
 return logger

def get_logger(name: str = "Mosaic") -> logging.Logger:
 """"""
 return logging.getLogger(name)

# ==================== ====================

def set_seed(seed: int = 42):
 """
 
 
 Args:
 seed: 
 """
 random.seed(seed)
 np.random.seed(seed)
 torch.manual_seed(seed)
 
 if torch.cuda.is_available():
 torch.cuda.manual_seed(seed)
 torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic = True
 torch.backends.cudnn.benchmark = False

# ==================== ====================

def get_device(device: Optional[str] = None) -> torch.device:
 """
 
 
 Args:
 device: (None)
 
 Returns:
 device: torch
 """
 if device is None:
 if torch.cuda.is_available():
 device = "cuda"
 else:
 device = "cpu"
 
 return torch.device(device)

def get_num_gpus() -> int:
 """GPU"""
 if torch.cuda.is_available():
 return torch.cuda.device_count()
 return 0

def get_gpu_memory_info() -> List[Dict[str, int]]:
 """GPU"""
 info = []
 
 if torch.cuda.is_available():
 for i in range(torch.cuda.device_count()):
 props = torch.cuda.get_device_properties(i)
 allocated = torch.cuda.memory_allocated(i)
 reserved = torch.cuda.memory_reserved(i)
 
 info.append({
 "device": i,
 "name": props.name,
 "total_memory": props.total_memory,
 "allocated_memory": allocated,
 "reserved_memory": reserved,
 "free_memory": props.total_memory - reserved,
 })
 
 return info

def print_gpu_info():
 """GPU"""
 if not torch.cuda.is_available():
 print("No GPU available")
 return
 
 print(f"Number of GPUs: {torch.cuda.device_count()}")
 
 for info in get_gpu_memory_info():
 print(f"\nGPU {info['device']}: {info['name']}")
 print(f" Total Memory: {info['total_memory'] / 1024**3:.2f} GB")
 print(f" Allocated: {info['allocated_memory'] / 1024**3:.2f} GB")
 print(f" Free: {info['free_memory'] / 1024**3:.2f} GB")

# ==================== ====================

def count_parameters(model: torch.nn.Module) -> int:
 """"""
 return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model: torch.nn.Module) -> int:
 """"""
 return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model: torch.nn.Module, input_size: Optional[tuple] = None):
 """"""
 print(f"Model: {model.__class__.__name__}")
 print(f"Total Parameters: {count_parameters(model):,}")
 print(f"Trainable Parameters: {count_trainable_parameters(model):,}")
 print(f"Non-trainable Parameters: {count_parameters(model) - count_trainable_parameters(model):,}")
 
 # 
 print("\nLayer Structure:")
 for name, module in model.named_children():
 num_params = sum(p.numel() for p in module.parameters())
 print(f" {name}: {module.__class__.__name__} ({num_params:,} params)")

def save_model(
 model: torch.nn.Module,
 save_path: str,
 optimizer: Optional[torch.optim.Optimizer] = None,
 epoch: Optional[int] = None,
 additional_info: Optional[Dict] = None,
):
 """
 
 
 Args:
 model: 
 save_path: 
 optimizer: 
 epoch: epoch
 additional_info: 
 """
 os.makedirs(os.path.dirname(save_path), exist_ok=True)
 
 checkpoint = {
 "model_state_dict": model.state_dict(),
 }
 
 if optimizer is not None:
 checkpoint["optimizer_state_dict"] = optimizer.state_dict()
 
 if epoch is not None:
 checkpoint["epoch"] = epoch
 
 if additional_info is not None:
 checkpoint.update(additional_info)
 
 torch.save(checkpoint, save_path)

def load_model(
 model: torch.nn.Module,
 load_path: str,
 optimizer: Optional[torch.optim.Optimizer] = None,
 device: Optional[torch.device] = None,
) -> Dict:
 """
 
 
 Args:
 model: 
 load_path: 
 optimizer: 
 device: 
 
 Returns:
 checkpoint: 
 """
 if device is None:
 device = get_device()
 
 checkpoint = torch.load(load_path, map_location=device)
 
 model.load_state_dict(checkpoint["model_state_dict"])
 
 if optimizer is not None and "optimizer_state_dict" in checkpoint:
 optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
 
 return checkpoint

# ==================== ====================

def ensure_dir(path: str):
 """"""
 Path(path).mkdir(parents=True, exist_ok=True)

def save_json(data: Dict, path: str):
 """JSON"""
 ensure_dir(os.path.dirname(path))
 with open(path, 'w', encoding='utf-8') as f:
 json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(path: str) -> Dict:
 """JSON"""
 with open(path, 'r', encoding='utf-8') as f:
 return json.load(f)

def get_timestamp() -> str:
 """"""
 return datetime.now().strftime('%Y%m%d_%H%M%S')

# ==================== ====================

class AverageMeter:
 """"""
 
 def __init__(self, name: str = ""):
 self.name = name
 self.reset()
 
 def reset(self):
 self.val = 0
 self.avg = 0
 self.sum = 0
 self.count = 0
 
 def update(self, val: float, n: int = 1):
 self.val = val
 self.sum += val * n
 self.count += n
 self.avg = self.sum / self.count
 
 def __str__(self):
 return f"{self.name}: {self.avg:.4f}"

class Timer:
 """"""
 
 def __init__(self):
 self.start_time = None
 self.elapsed = 0
 
 def start(self):
 self.start_time = datetime.now()
 
 def stop(self):
 if self.start_time is not None:
 self.elapsed = (datetime.now() - self.start_time).total_seconds()
 self.start_time = None
 return self.elapsed
 
 def __enter__(self):
 self.start()
 return self
 
 def __exit__(self, *args):
 self.stop()
 
 def __str__(self):
 return f"{self.elapsed:.2f}s"

# ==================== ====================

def plot_training_curves(
 history: Dict[str, List[float]],
 save_path: Optional[str] = None,
 title: str = "Training Curves",
):
 """
 
 
 Args:
 history: {metric_name: [values]}
 save_path: 
 title: 
 """
 try:
 import matplotlib.pyplot as plt
 except ImportError:
 print("matplotlib not available for plotting")
 return
 
 num_metrics = len(history)
 fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4))
 
 if num_metrics == 1:
 axes = [axes]
 
 for ax, (metric_name, values) in zip(axes, history.items()):
 epochs = range(1, len(values) + 1)
 ax.plot(epochs, values, marker='o', markersize=3)
 ax.set_xlabel('Epoch')
 ax.set_ylabel(metric_name)
 ax.set_title(metric_name)
 ax.grid(True, alpha=0.3)
 
 plt.suptitle(title)
 plt.tight_layout()
 
 if save_path:
 ensure_dir(os.path.dirname(save_path))
 plt.savefig(save_path, dpi=150, bbox_inches='tight')
 
 plt.close()

def plot_roc_curve(
 y_true: np.ndarray,
 y_scores: np.ndarray,
 save_path: Optional[str] = None,
 label: str = "Model",
):
 """
 ROC
 
 Args:
 y_true: 
 y_scores: 
 save_path: 
 label: 
 """
 try:
 import matplotlib.pyplot as plt
 from sklearn.metrics import roc_curve, auc
 except ImportError:
 print("Required libraries not available for plotting")
 return
 
 fpr, tpr, _ = roc_curve(y_true, y_scores)
 roc_auc = auc(fpr, tpr)
 
 plt.figure(figsize=(8, 6))
 plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.4f})')
 plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
 
 plt.xlim([0.0, 1.0])
 plt.ylim([0.0, 1.05])
 plt.xlabel('False Positive Rate')
 plt.ylabel('True Positive Rate')
 plt.title('ROC Curve')
 plt.legend(loc='lower right')
 plt.grid(True, alpha=0.3)
 
 if save_path:
 ensure_dir(os.path.dirname(save_path))
 plt.savefig(save_path, dpi=150, bbox_inches='tight')
 
 plt.close()

if __name__ == "__main__":
 # 
 print("Testing Utility Functions...")
 
 # 
 print("\nTesting set_seed...")
 set_seed(42)
 print(f"Random int: {random.randint(0, 100)}")
 
 # 
 print("\nTesting device functions...")
 device = get_device()
 print(f"Device: {device}")
 print(f"Number of GPUs: {get_num_gpus()}")
 
 # 
 print("\nTesting logging...")
 logger = setup_logging(log_level="INFO")
 logger.info("Test log message")
 
 # 
 print("\nTesting Timer...")
 import time
 with Timer() as timer:
 time.sleep(0.1)
 print(f"Elapsed time: {timer}")
 
 # 
 print("\nTesting AverageMeter...")
 meter = AverageMeter("loss")
 for i in range(5):
 meter.update(i * 0.1)
 print(meter)
 
 print("\nAll tests passed!")
