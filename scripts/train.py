#!/usr/bin/env python
"""Main training script for the AI-Math-Reasoning framework."""

import os
import json
import argparse
import logging
from typing import Dict, Any

import torch
import numpy as np

from ai_math_reasoning.models import create_model
from ai_math_reasoning.grpo.trainer import GRPOTrainer
from ai_math_reasoning.distillation.trainer import DistillationTrainer
from ai_math_reasoning.prm.reranker import PRMReranker
from ai_math_reasoning.data.datasets.math_dataset import MathReasoningDataset, HuggingFaceMathDataset
from ai_math_reasoning.utils.data_utils import load_math_dataset, create_training_splits, create_dataloaders
from ai_math_reasoning.utils.training_utils import (
    set_seed, save_config, load_config, initialize_wandb, 
    initialize_tensorboard, log_metrics
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train models for mathematical reasoning")
    
    # Training mode
    parser.add_argument(
        "--mode", 
        type=str, 
        default="grpo", 
        choices=["grpo", "prm", "distillation", "sft"],
        help="Training mode"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="qwen",
        choices=["qwen", "deepseek", "transformer"],
        help="Type of model to use"
    )
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default=None,
        help="Path to pre-trained model or model identifier"
    )
    
    # For distillation, we need teacher and student models
    parser.add_argument(
        "--teacher_model_type", 
        type=str, 
        default="deepseek",
        help="Type of teacher model for distillation"
    )
    parser.add_argument(
        "--teacher_model_path", 
        type=str, 
        default=None,
        help="Path to teacher model for distillation"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="numina/numina-math-filtered",
        help="Path to dataset or HuggingFace dataset identifier"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="train",
        help="Dataset split to use for training"
    )
    parser.add_argument(
        "--eval_split", 
        type=str, 
        default="validation",
        help="Dataset split to use for validation"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None,
        help="Maximum number of samples to use"
    )
    parser.add_argument(
        "--eval_ratio", 
        type=float, 
        default=0.1,
        help="Ratio of training data to use for validation"
    )
    
    # Training arguments
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=None,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=5e-6,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--warmup_steps", 
        type=int, 
        default=100,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--max_grad_norm", 
        type=float, 
        default=1.0,
        help="Maximum gradient norm"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    
    # GRPO specific arguments
    parser.add_argument(
        "--group_size", 
        type=int, 
        default=8,
        help="Group size for GRPO"
    )
    parser.add_argument(
        "--kl_coef", 
        type=float, 
        default=0.05,
        help="KL coefficient for GRPO"
    )
    parser.add_argument(
        "--clip_range", 
        type=float, 
        default=0.2,
        help="Clipping range for GRPO"
    )
    
    # Distillation specific arguments
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=2.0,
        help="Temperature for distillation"
    )
    parser.add_argument(
        "--alpha", 
        type=float, 
        default=0.5,
        help="Alpha for distillation"
    )
    
    # Logging arguments
    parser.add_argument(
        "--use_wandb", 
        action="store_true",
        help="Whether to use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb_project", 
        type=str, 
        default="ai-math-reasoning",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_name", 
        type=str, 
        default=None,
        help="Weights & Biases run name"
    )
    parser.add_argument(
        "--use_tensorboard", 
        action="store_true",
        help="Whether to use TensorBoard for logging"
    )
    parser.add_argument(
        "--log_interval", 
        type=int, 
        default=10,
        help="Logging interval"
    )
    parser.add_argument(
        "--eval_interval", 
        type=int, 
        default=100,
        help="Evaluation interval"
    )
    parser.add_argument(
        "--save_interval", 
        type=int, 
        default=500,
        help="Checkpoint saving interval"
    )
    
    # Model optimization arguments
    parser.add_argument(
        "--precision", 
        type=str, 
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Precision for model weights"
    )
    parser.add_argument(
        "--use_lora", 
        action="store_true",
        help="Whether to use LoRA for fine-tuning"
    )
    parser.add_argument(
        "--lora_r", 
        type=int, 
        default=16,
        help="LoRA r parameter"
    )
    parser.add_argument(
        "--lora_alpha", 
        type=int, 
        default=32,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout", 
        type=float, 
        default=0.05,
        help="LoRA dropout"
    )
    parser.add_argument(
        "--use_amp", 
        action="store_true",
        help="Whether to use automatic mixed precision"
    )
    
    return parser.parse_args()


def load_configuration(args) -> Dict[str, Any]:
    """Load configuration from file or command-line arguments."""
    # Start with default configuration
    config = {
        "mode": args.mode,
        "model_type": args.model_type,
        "model_name_or_path": args.model_name_or_path,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "eval_split": args.eval_split,
        "max_samples": args.max_samples,
        "eval_ratio": args.eval_ratio,
        "output_dir": args.output_dir,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size or args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "log_interval": args.log_interval,
        "eval_interval": args.eval_interval,
        "save_interval": args.save_interval,
        "precision": args.precision,
        "use_lora": args.use_lora,
        "lora_config": {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        },
        "use_amp": args.use_amp,
    }
    
    # Add mode-specific config
    if args.mode == "grpo":
        config.update({
            "group_size": args.group_size,
            "kl_coef": args.kl_coef,
            "clip_range": args.clip_range,
        })
    elif args.mode == "distillation":
        config.update({
            "teacher_model_type": args.teacher_model_type,
            "teacher_model_path": args.teacher_model_path,
            "temperature": args.temperature,
            "alpha": args.alpha,
        })
    
    # Override with config file if provided
    if args.config is not None:
        with open(args.config, "r") as f:
            file_config = json.load(f)
            config.update(file_config)
    
    return config


def train_grpo(config: Dict[str, Any]) -> None:
    """Train a model using GRPO.
    
    Args:
        config: Training configuration
    """
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Save configuration
    save_config(config, os.path.join(config["output_dir"], "config.json"))
    
    # Load dataset
    logging.info(f"Loading dataset: {config['dataset_path']}")
    dataset = load_math_dataset(
        dataset_path=config["dataset_path"],
        split=config["split"],
        max_samples=config["max_samples"],
        seed=config["seed"]
    )
    
    # Create training and evaluation splits
    train_dataset, eval_dataset = create_training_splits(
        dataset=dataset,
        train_ratio=1.0 - config["eval_ratio"],
        seed=config["seed"]
    )
    
    logging.info(f"Training examples: {len(train_dataset)}")
    logging.info(f"Evaluation examples: {len(eval_dataset)}")
    
    # Create model
    logging.info(f"Creating model: {config['model_type']}")
    model = create_model(
        model_type=config["model_type"],
        model_name_or_path=config["model_name_or_path"],
        precision=config["precision"],
        use_lora=config["use_lora"],
        lora_config=config["lora_config"] if config["use_lora"] else None
    )
    
    # Create reward model (optional)
    reward_model = None
    if config.get("reward_model_path") is not None:
        logging.info(f"Creating reward model")
        reward_model = create_model(
            model_type=config.get("reward_model_type", config["model_type"]),
            model_name_or_path=config["reward_model_path"],
            precision=config["precision"]
        )
    
    # Create reference model (optional)
    reference_model = None
    if config.get("reference_model_path") is not None:
        logging.info(f"Creating reference model")
        reference_model = create_model(
            model_type=config.get("reference_model_type", config["model_type"]),
            model_name_or_path=config["reference_model_path"],
            precision=config["precision"]
        )
    
    # Create trainer
    logging.info(f"Creating GRPO trainer")
    trainer = GRPOTrainer(
        policy_model=model,
        reward_model=reward_model,
        reference_model=reference_model,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        kl_coef=config["kl_coef"],
        clip_range=config["clip_range"],
        group_size=config["group_size"],
        max_grad_norm=config["max_grad_norm"],
        use_amp=config["use_amp"]
    )
    
    # Initialize logging
    wandb_enabled = False
    tb_writer = None
    
    if config.get("use_wandb", False):
        wandb_enabled = initialize_wandb(
            project_name=config.get("wandb_project", "ai-math-reasoning"),
            experiment_name=config.get("wandb_name", f"grpo-{config['model_type']}"),
            config=config
        )
    
    if config.get("use_tensorboard", False):
        tb_writer = initialize_tensorboard(
            log_dir=os.path.join(config["output_dir"], "logs"),
            experiment_name=f"grpo-{config['model_type']}"
        )
    
    # Define callback function for logging metrics
    def log_callback(trainer, stats, episode, batch_idx=None):
        # Extract metrics
        if batch_idx is not None:
            step = episode * len(train_dataset) // config["batch_size"] + batch_idx
            metrics = {
                "train/loss": stats["loss"][-1],
                "train/reward": stats["reward"][-1],
                "train/kl_div": stats["kl_div"][-1],
            }
        else:
            step = episode
            metrics = {
                "train/loss": np.mean(stats["loss"]),
                "train/reward": np.mean(stats["reward"]),
                "train/kl_div": np.mean(stats["kl_div"]),
            }
        
        # Add eval metrics if available
        if batch_idx is not None and "eval_loss" in stats and len(stats["eval_loss"]) > 0:
            metrics.update({
                "eval/loss": stats["eval_loss"][-1],
                "eval/reward": stats["eval_reward"][-1] if "eval_reward" in stats else 0.0,
                "eval/kl_div": stats["eval_kl_div"][-1] if "eval_kl_div" in stats else 0.0,
            })
        
        # Log metrics
        log_metrics(
            metrics=metrics,
            step=step,
            wandb_enabled=wandb_enabled,
            tb_writer=tb_writer
        )
    
    # Start training
    logging.info(f"Starting GRPO training for {config['num_epochs']} epochs")
    
    # Train the model
    stats = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_episodes=config["num_epochs"],
        batch_size=config["batch_size"],
        log_interval=config["log_interval"],
        eval_interval=config["eval_interval"],
        save_interval=config["save_interval"],
        output_dir=config["output_dir"],
        callback=log_callback
    )
    
    # Save final metrics
    final_metrics = {
        "loss": np.mean(stats["loss"]),
        "reward": np.mean(stats["reward"]),
        "kl_div": np.mean(stats["kl_div"]),
    }
    
    with open(os.path.join(config["output_dir"], "metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    logging.info(f"Training completed. Metrics saved to {config['output_dir']}/metrics.json")


def train_distillation(config: Dict[str, Any]) -> None:
    """Train a model using knowledge distillation.
    
    Args:
        config: Training configuration
    """
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Save configuration
    save_config(config, os.path.join(config["output_dir"], "config.json"))
    
    # Create teacher model
    logging.info(f"Creating teacher model: {config['teacher_model_type']}")
    teacher_model = create_model(
        model_type=config["teacher_model_type"],
        model_name_or_path=config["teacher_model_path"],
        precision=config["precision"]
    )
    
    # Create student model
    logging.info(f"Creating student model: {config['model_type']}")
    student_model = create_model(
        model_type=config["model_type"],
        model_name_or_path=config["model_name_or_path"],
        precision=config["precision"],
        use_lora=config["use_lora"],
        lora_config=config["lora_config"] if config["use_lora"] else None
    )
    
    # Load dataset with tokenizer
    logging.info(f"Loading dataset: {config['dataset_path']}")
    dataset = load_math_dataset(
        dataset_path=config["dataset_path"],
        split=config["split"],
        tokenizer=student_model.tokenizer,
        max_samples=config["max_samples"],
        seed=config["seed"]
    )
    
    # Create training and evaluation splits
    train_dataset, eval_dataset = create_training_splits(
        dataset=dataset,
        train_ratio=1.0 - config["eval_ratio"],
        seed=config["seed"]
    )
    
    logging.info(f"Training examples: {len(train_dataset)}")
    logging.info(f"Evaluation examples: {len(eval_dataset)}")
    
    # Create dataloader
    train_loader, eval_loader = create_dataloaders(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=config["batch_size"],
        eval_batch_size=config["eval_batch_size"]
    )
    
    # Create trainer
    logging.info(f"Creating distillation trainer")
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=config["temperature"],
        alpha=config["alpha"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"],
        use_amp=config["use_amp"]
    )
    
    # Initialize logging
    wandb_enabled = False
    tb_writer = None
    
    if config.get("use_wandb", False):
        wandb_enabled = initialize_wandb(
            project_name=config.get("wandb_project", "ai-math-reasoning"),
            experiment_name=config.get("wandb_name", f"distillation-{config['model_type']}"),
            config=config
        )
    
    if config.get("use_tensorboard", False):
        tb_writer = initialize_tensorboard(
            log_dir=os.path.join(config["output_dir"], "logs"),
            experiment_name=f"distillation-{config['model_type']}"
        )
    
    # Define callback function for logging metrics
    def log_callback(trainer, stats, epoch, batch_idx=None):
        # Extract metrics
        if batch_idx is not None:
            step = epoch * len(train_dataset) // config["batch_size"] + batch_idx
            metrics = {
                "train/loss": stats["loss"][-1],
                "train/kl_loss": stats["kl_loss"][-1],
                "train/ce_loss": stats["ce_loss"][-1],
            }
        else:
            step = epoch
            metrics = {
                "train/loss": np.mean(stats["loss"]),
                "train/kl_loss": np.mean(stats["kl_loss"]),
                "train/ce_loss": np.mean(stats["ce_loss"]),
            }
        
        # Add eval metrics if available
        if batch_idx is not None and "eval_loss" in stats and len(stats["eval_loss"]) > 0:
            metrics.update({
                "eval/loss": stats["eval_loss"][-1],
                "eval/kl_loss": stats["eval_kl_loss"][-1],
                "eval/ce_loss": stats["eval_ce_loss"][-1],
            })
        
        # Log metrics
        log_metrics(
            metrics=metrics,
            step=step,
            wandb_enabled=wandb_enabled,
            tb_writer=tb_writer
        )
    
    # Start training
    logging.info(f"Starting distillation training for {config['num_epochs']} epochs")
    
    # Train the model
    stats = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=config["batch_size"],
        num_epochs=config["num_epochs"],
        log_interval=config["log_interval"],
        eval_interval=config["eval_interval"],
        save_interval=config["save_interval"],
        output_dir=config["output_dir"],
        callback=log_callback
    )
    
    # Save final metrics
    final_metrics = {
        "loss": np.mean(stats["loss"]),
        "kl_loss": np.mean(stats["kl_loss"]),
        "ce_loss": np.mean(stats["ce_loss"]),
    }
    
    with open(os.path.join(config["output_dir"], "metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    logging.info(f"Training completed. Metrics saved to {config['output_dir']}/metrics.json")


def train_prm(config: Dict[str, Any]) -> None:
    """Train a Preference Reward Model.
    
    Args:
        config: Training configuration
    """
    # TODO: Implement PRM training
    # This would be similar to the other training functions, but using the
    # PRM-specific dataset and trainer
    raise NotImplementedError("PRM training not implemented yet")


def train_sft(config: Dict[str, Any]) -> None:
    """Train a model using supervised fine-tuning.
    
    Args:
        config: Training configuration
    """
    # TODO: Implement SFT training
    # This would be a simpler version of the other training functions, using
    # standard supervised learning
    raise NotImplementedError("SFT training not implemented yet")


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w')
        ] if os.path.exists(args.output_dir) else [logging.StreamHandler()]
    )
    
    # Load configuration
    config = load_configuration(args)
    
    # Set random seed
    set_seed(config["seed"])
    
    # Train model based on mode
    if config["mode"] == "grpo":
        train_grpo(config)
    elif config["mode"] == "distillation":
        train_distillation(config)
    elif config["mode"] == "prm":
        train_prm(config)
    elif config["mode"] == "sft":
        train_sft(config)
    else:
        raise ValueError(f"Unknown training mode: {config['mode']}")


if __name__ == "__main__":
    main()
