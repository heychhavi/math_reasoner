"""Utilities for model training and evaluation."""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import tensorboard as tb
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # Set deterministic behavior for CUDNN
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.01,
    no_decay_keywords: Tuple[str, ...] = ("bias", "LayerNorm.weight")
) -> List[Dict[str, Any]]:
    """Get parameter groups for optimizer with weight decay.
    
    Args:
        model: PyTorch model
        weight_decay: Weight decay value
        no_decay_keywords: Parameter names which should not use weight decay
        
    Returns:
        List of parameter groups for optimizer
    """
    # Get parameter names
    param_names = [name for name, _ in model.named_parameters()]
    
    # Create parameter groups with and without weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if any(keyword in name for keyword in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    # Create parameter groups
    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        }
    ]
    
    return optimizer_grouped_parameters


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute the gradient norm for a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    """A simple timer class for measuring execution time."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.start_time = time.time()
        self.end_time = None
        
    def stop(self):
        self.end_time = time.time()
        
    @property
    def elapsed(self):
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


class LinearWarmupWithDecay(_LRScheduler):
    """Scheduler with linear warmup and cosine decay.
    
    This scheduler linearly increases the learning rate from 0 to the initial
    learning rate over the first warmup_steps, and then decays the learning
    rate using a cosine schedule.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1
    ):
        """Initialize the scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr_ratio: Minimum learning rate ratio
            last_epoch: Last epoch
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Get learning rate based on current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            lr_mult = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [base_lr * lr_mult for base_lr in self.base_lrs]
        else:
            # Cosine decay after warmup
            progress = float(self.last_epoch - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            decay_factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay
            return [base_lr * decay_factor for base_lr in self.base_lrs]


def initialize_wandb(
    project_name: str,
    experiment_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """Initialize Weights & Biases for experiment tracking.
    
    Args:
        project_name: W&B project name
        experiment_name: W&B experiment name (optional)
        config: Configuration for the experiment (optional)
        
    Returns:
        True if initialization was successful, False otherwise
    """
    if not WANDB_AVAILABLE:
        logging.warning("wandb is not installed. Skipping W&B initialization.")
        return False
    
    try:
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=config
        )
        return True
    except Exception as e:
        logging.warning(f"Failed to initialize wandb: {str(e)}")
        return False


def initialize_tensorboard(
    log_dir: str,
    experiment_name: Optional[str] = None
) -> Optional[SummaryWriter]:
    """Initialize TensorBoard for experiment tracking.
    
    Args:
        log_dir: Directory for TensorBoard logs
        experiment_name: Experiment name (optional)
        
    Returns:
        TensorBoard SummaryWriter or None if initialization failed
    """
    if not TB_AVAILABLE:
        logging.warning("tensorboard is not installed. Skipping TensorBoard initialization.")
        return None
    
    try:
        # Create log directory
        if experiment_name:
            log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(log_dir, exist_ok=True)
        
        writer = SummaryWriter(log_dir)
        return writer
    except Exception as e:
        logging.warning(f"Failed to initialize TensorBoard: {str(e)}")
        return None


def log_metrics(
    metrics: Dict[str, float],
    step: int,
    wandb_enabled: bool = False,
    tb_writer: Optional[SummaryWriter] = None,
    print_metrics: bool = True
) -> None:
    """Log metrics to W&B, TensorBoard, and terminal.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Current step
        wandb_enabled: Whether W&B is enabled
        tb_writer: TensorBoard writer (optional)
        print_metrics: Whether to print metrics to terminal
    """
    # Log to W&B
    if wandb_enabled and WANDB_AVAILABLE:
        wandb.log(metrics, step=step)
    
    # Log to TensorBoard
    if tb_writer is not None:
        for name, value in metrics.items():
            tb_writer.add_scalar(name, value, step)
    
    # Print metrics
    if print_metrics:
        metrics_str = " | ".join([f"{name}: {value:.4f}" for name, value in metrics.items()])
        logging.info(f"Step {step} - {metrics_str}")


def create_checkpoint(
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    epoch: int = 0,
    step: int = 0,
    best_metric: Optional[float] = None,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a checkpoint for model, optimizer, and other training state.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        epoch: Current epoch
        step: Current step
        best_metric: Best metric value (optional)
        metrics: Dictionary of metrics (optional)
        config: Configuration (optional)
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "step": step,
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
    if best_metric is not None:
        checkpoint["best_metric"] = best_metric
        
    if metrics is not None:
        checkpoint["metrics"] = metrics
        
    if config is not None:
        checkpoint["config"] = config
        
    return checkpoint


def save_checkpoint(
    checkpoint: Dict[str, Any],
    save_path: str,
    is_best: bool = False,
    best_path: Optional[str] = None
) -> None:
    """Save a checkpoint to disk.
    
    Args:
        checkpoint: Checkpoint dictionary
        save_path: Path to save the checkpoint
        is_best: Whether this is the best checkpoint
        best_path: Path to save the best checkpoint (optional)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, save_path)
    logging.info(f"Saved checkpoint to {save_path}")
    
    # Save best checkpoint
    if is_best and best_path is not None:
        os.makedirs(os.path.dirname(best_path), exist_ok=True)
        torch.save(checkpoint, best_path)
        logging.info(f"Saved best checkpoint to {best_path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: Optional[torch.device] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """Load a checkpoint from disk.
    
    Args:
        path: Path to the checkpoint
        model: PyTorch model
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        device: Device to load to (optional)
        strict: Whether to strictly enforce that the keys in state_dict match
            the keys returned by this module's state_dict() function
            
    Returns:
        Checkpoint dictionary
    """
    # Load checkpoint
    if device is None:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    
    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
    return checkpoint


def save_config(config: Dict[str, Any], path: str) -> None:
    """Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        path: Path to save the configuration
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save configuration
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file.
    
    Args:
        path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    progress_bar: bool = True
) -> Dict[str, float]:
    """Evaluate a model on a dataset.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation data
        criterion: Loss function (optional)
        device: Device to evaluate on (optional)
        progress_bar: Whether to show a progress bar
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Set model to evaluation mode
    model.eval()
    
    # Create meters for tracking metrics
    loss_meter = AverageMeter()
    
    # Create progress bar
    if progress_bar:
        pbar = tqdm(total=len(data_loader), desc="Evaluating")
    
    # Evaluate on batches
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            if device is not None:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            
            # Calculate loss
            if criterion is not None:
                loss = criterion(outputs, batch)
                loss_meter.update(loss.item())
            
            # Update progress bar
            if progress_bar:
                pbar.update(1)
    
    # Close progress bar
    if progress_bar:
        pbar.close()
    
    # Calculate metrics
    metrics = {"loss": loss_meter.avg}
    
    # Restore model to training mode
    model.train()
    
    return metrics
