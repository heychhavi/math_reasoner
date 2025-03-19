#!/usr/bin/env python
"""
Train Preference Reward Model (PRM) for mathematical reasoning.

This script trains a reward model to rank mathematical solutions based on quality,
which can be used for reranking solutions in the inference pipeline.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path to import from ai_math_reasoning
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_math_reasoning.models import create_model
from ai_math_reasoning.prm.reranker import PRMReranker
from ai_math_reasoning.utils.training_utils import setup_logging, set_seed


class PRMDataset(Dataset):
    """Dataset for PRM training with preferred and non-preferred solutions."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
    ):
        """Initialize the PRM dataset.
        
        Args:
            data_path: Path to data file with solution pairs
            tokenizer: Tokenizer for tokenization
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.pairs = json.load(f)
        
        logging.info(f"Loaded {len(self.pairs)} preference pairs from {data_path}")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        
        problem = pair["problem"]
        better_solution = pair["better_solution"]
        worse_solution = pair["worse_solution"]
        
        # Tokenize better solution
        better_input = f"Problem: {problem}\nSolution: {better_solution}"
        better_tokenized = self.tokenizer(
            better_input,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Tokenize worse solution
        worse_input = f"Problem: {problem}\nSolution: {worse_solution}"
        worse_tokenized = self.tokenizer(
            worse_input,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "better_input_ids": better_tokenized["input_ids"][0],
            "better_attention_mask": better_tokenized["attention_mask"][0],
            "worse_input_ids": worse_tokenized["input_ids"][0],
            "worse_attention_mask": worse_tokenized["attention_mask"][0],
        }


def create_or_load_prm_dataset(
    config: Dict[str, Any],
    tokenizer,
    split: str = "train"
) -> PRMDataset:
    """Create or load the PRM dataset.
    
    Args:
        config: Configuration dictionary
        tokenizer: Tokenizer for tokenization
        split: Dataset split ('train' or 'eval')
        
    Returns:
        PRMDataset
    """
    dataset_config = config["dataset"]
    
    # Determine path based on split
    if split == "train":
        data_path = dataset_config["train_dataset_path"]
    else:
        data_path = dataset_config["eval_dataset_path"]
    
    # Create dataset
    dataset = PRMDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=config["model"]["max_length"]
    )
    
    return dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PRM for mathematical reasoning")
    
    # Config
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/prm/default_config.json",
        help="Path to configuration file"
    )
    
    # Model options
    parser.add_argument(
        "--model_type", 
        type=str, 
        help="Type of model to use (overrides config)"
    )
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        help="Path to the pre-trained model (overrides config)"
    )
    
    # Output options
    parser.add_argument(
        "--output_dir", 
        type=str, 
        help="Output directory (overrides config)"
    )
    
    # Training options
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        help="Random seed (overrides config)"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


def update_config_with_args(config: Dict[str, Any], args) -> Dict[str, Any]:
    """Update configuration with command line arguments.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        Updated configuration dictionary
    """
    # Override model settings
    if args.model_type:
        config["model"]["model_type"] = args.model_type
    
    if args.model_name_or_path:
        config["model"]["model_name_or_path"] = args.model_name_or_path
    
    # Override output settings
    if args.output_dir:
        config["output"]["output_dir"] = args.output_dir
    
    # Override training settings
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    
    if args.num_epochs:
        config["training"]["num_epochs"] = args.num_epochs
    
    if args.seed:
        config["training"]["seed"] = args.seed
    
    return config


def train_prm(config: Dict[str, Any]):
    """Train the PRM model.
    
    Args:
        config: Configuration dictionary
    """
    # Set random seed
    set_seed(config["training"]["seed"])
    
    # Create output directory
    os.makedirs(config["output"]["output_dir"], exist_ok=True)
    os.makedirs(config["output"]["logging_dir"], exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(config["output"]["logging_dir"], "train_prm.log")
    setup_logging(log_file=log_file)
    
    # Log configuration
    logging.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Create base model
    model = create_model(
        model_type=config["model"]["model_type"],
        model_name_or_path=config["model"]["model_name_or_path"],
        precision=config["model"]["precision"],
        max_length=config["model"]["max_length"],
        use_flash_attention=config["model"]["use_flash_attention"],
        use_lora=config["model"]["use_lora"],
        lora_config=config["model"].get("lora_config", None)
    )
    
    # Create PRM reranker
    prm = PRMReranker(
        base_model=model,
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        margin=config["training"]["margin"],
        beta=config["training"]["beta"],
        max_grad_norm=config["training"]["max_grad_norm"]
    )
    
    # Create datasets
    train_dataset = create_or_load_prm_dataset(
        config=config,
        tokenizer=model.tokenizer,
        split="train"
    )
    
    eval_dataset = create_or_load_prm_dataset(
        config=config,
        tokenizer=model.tokenizer,
        split="eval"
    ) if "eval_dataset_path" in config["dataset"] else None
    
    # Train the model
    logging.info("Starting PRM training...")
    stats = prm.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=config["training"]["batch_size"],
        num_epochs=config["training"]["num_epochs"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        log_interval=config["training"]["log_interval"],
        eval_interval=config["training"]["eval_interval"],
        save_interval=config["training"]["save_interval"],
        save_dir=config["output"]["output_dir"]
    )
    
    # Save training stats
    stats_path = os.path.join(config["output"]["output_dir"], "training_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        # Convert stats to JSON-serializable format
        serializable_stats = {}
        for k, v in stats.items():
            if isinstance(v, (list, int, float, str)):
                serializable_stats[k] = v
            elif isinstance(v, np.ndarray):
                serializable_stats[k] = v.tolist()
            else:
                serializable_stats[k] = str(v)
        
        json.dump(serializable_stats, f, indent=2)
    
    logging.info(f"Training completed. Model saved to {config['output']['output_dir']}")
    logging.info(f"Training stats saved to {stats_path}")


def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update configuration with command line arguments
    config = update_config_with_args(config, args)
    
    # Train PRM
    train_prm(config)


if __name__ == "__main__":
    main()
