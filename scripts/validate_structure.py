#!/usr/bin/env python
"""
Mosaic Framework - Structure Validator
====================================
Usage:
    python scripts/validate_structure.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

EXPECTED_STRUCTURE = {
    "configs": ["config.yaml"],
    "src": ["__init__.py"],
    "src/models": [
        "__init__.py",
        "expert_model.py",
        "router_network.py",
        "meta_learner.py",
        "attack_classifier.py",
    ],
    "src/data": [
        "__init__.py",
        "data_loader.py",
        "domain_clustering.py",
    ],
    "src/training": [
        "__init__.py",
        "expert_trainer.py",
        "ensemble_trainer.py",
        "attack_trainer.py",
    ],
    "src/evaluation": [
        "__init__.py",
        "metrics.py",
    ],
    "src/utils": [
        "__init__.py",
        "helpers.py",
    ],
    "scripts": [
        "train.py",
        "run_experiments.py",
        "demo.py",
    ],
    "tests": [
        "__init__.py",
        "test_models.py",
    ],
}

EXPECTED_CONTENTS = {
    "src/models/expert_model.py": [
        "class ExpertModel",
        "class ExpertModelSmall",
        "class MultiExpertSystem",
        "class RMSNorm",
        "class RotaryPositionalEmbedding",
    ],
    "src/models/router_network.py": [
        "class RouterNetwork",
        "class FeatureExtractor",
        "class SimpleFeatureExtractor",
    ],
    "src/models/meta_learner.py": [
        "class MetaLearner",
        "class MetaFeatureBuilder",
    ],
    "src/models/attack_classifier.py": [
        "class AttackClassifier",
        "class AttackFeatureExtractor",
        "class BaselineAttacks",
    ],
    "src/data/data_loader.py": [
        "class TextDataset",
        "class MembershipDataset",
        "class DataLoaderFactory",
    ],
    "src/data/domain_clustering.py": [
        "class DomainComplexityAnalyzer",
        "class DomainClusterer",
        "class DomainManager",
    ],
    "src/training/expert_trainer.py": [
        "class ExpertTrainer",
        "class MultiExpertTrainer",
        "class DistillationLoss",
    ],
    "src/training/ensemble_trainer.py": [
        "class EnsembleTrainer",
        "class RouterLoss",
    ],
    "src/training/attack_trainer.py": [
        "class AttackTrainer",
        "class EarlyStopping",
    ],
    "src/evaluation/metrics.py": [
        "class MembershipInferenceEvaluator",
        "class BaselineComparator",
        "class MultiGranularityEvaluator",
    ],
}


def check_file_exists(file_path: Path) -> bool:
    return file_path.exists()


def check_file_contains(file_path: Path, patterns: list) -> dict:
    results = {}
    
    if not file_path.exists():
        return {p: False for p in patterns}
    
    content = file_path.read_text(encoding='utf-8')
    
    for pattern in patterns:
        results[pattern] = pattern in content
        
    return results


def validate_structure():
    print("=" * 60)
    print("Mosaic Framework - Structure Validation")
    print("=" * 60)
    
    all_passed = True

    print("\n1. Checking Directory Structure...")
    print("-" * 40)
    
    for directory, files in EXPECTED_STRUCTURE.items():
        dir_path = PROJECT_ROOT / directory
        
        if not dir_path.exists():
            print(f"   [MISSING] Directory: {directory}/")
            all_passed = False
            continue
            
        for file_name in files:
            file_path = dir_path / file_name
            
            if file_path.exists():
                print(f"   [OK] {directory}/{file_name}")
            else:
                print(f"   [MISSING] {directory}/{file_name}")
                all_passed = False

    print("\n2. Checking File Contents...")
    print("-" * 40)
    
    for file_rel_path, patterns in EXPECTED_CONTENTS.items():
        file_path = PROJECT_ROOT / file_rel_path
        
        if not file_path.exists():
            print(f"   [SKIP] {file_rel_path} (file not found)")
            continue
            
        results = check_file_contains(file_path, patterns)
        missing = [p for p, found in results.items() if not found]
        
        if not missing:
            print(f"   [OK] {file_rel_path} ({len(patterns)} patterns)")
        else:
            print(f"   [PARTIAL] {file_rel_path}")
            for pattern in missing:
                print(f"       - Missing: {pattern}")
            all_passed = False

    print("\n3. Checking Configuration...")
    print("-" * 40)
    
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    if config_path.exists():
        content = config_path.read_text()
        required_sections = [
            "expert_system",
            "domain_training",
            "ensemble_training",
            "attack_execution",
        ]
        
        for section in required_sections:
            if section in content:
                print(f"   [OK] Config section: {section}")
            else:
                print(f"   [MISSING] Config section: {section}")
                all_passed = False
    else:
        print("   [MISSING] config.yaml")
        all_passed = False

    print("\n4. Checking Requirements...")
    print("-" * 40)
    
    req_path = PROJECT_ROOT / "requirements.txt"
    if req_path.exists():
        content = req_path.read_text()
        required_packages = ["torch", "transformers", "numpy", "scikit-learn"]
        
        for package in required_packages:
            if package in content:
                print(f"   [OK] Dependency: {package}")
            else:
                print(f"   [MISSING] Dependency: {package}")
                all_passed = False
    else:
        print("   [MISSING] requirements.txt")
        all_passed = False

    print("\n" + "=" * 60)

    total_files = 0
    total_lines = 0
    total_size = 0
    
    for ext in ['.py', '.yaml', '.txt', '.md']:
        for file_path in PROJECT_ROOT.rglob(f'*{ext}'):
            if '__pycache__' not in str(file_path):
                total_files += 1
                total_size += file_path.stat().st_size
                try:
                    total_lines += len(file_path.read_text().splitlines())
                except:
                    pass
                    
    print(f"\nProject Statistics:")
    print(f"   Total files: {total_files}")
    print(f"   Total lines: {total_lines:,}")
    print(f"   Total size: {total_size / 1024:.1f} KB")

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All validation checks PASSED!")
    else:
        print("✗ Some validation checks FAILED!")
        print("  Please review the [MISSING] items above.")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = validate_structure()
    sys.exit(0 if success else 1)
