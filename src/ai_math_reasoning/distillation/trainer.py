"""Implementation of knowledge distillation for mathematical reasoning."""

import os
import json
import time
import logging
import random
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

from ai_math_reasoning.models import MathReasoningModel


class DistillationTrainer:
    """Trainer for knowledge distillation.
    
    This class implements knowledge distillation from a teacher model to a 
    student model for mathematical reasoning tasks.
    """
    
    def __init__(
        self,
        teacher_model: MathReasoningModel,
        student_model: MathReasoningModel,
        temperature: float = 2.0,
        alpha: float = 0.5,
        learning_rate: float = 5e-6,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        use_amp: bool = False
    ):
        """Initialize a distillation trainer.
        
        Args:
            teacher_model: Teacher model
            student_model: Student model to train
            temperature: Temperature for softening logits
            alpha: Weight for distillation loss (1-alpha for task loss)
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            max_grad_norm: Maximum gradient norm for clipping
            use_amp: Whether to use automatic mixed precision
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and AMP_AVAILABLE
        
        # Prepare models
        self.teacher_model.model.eval()  # Teacher model in evaluation mode
        self.student_model.prepare_for_training()  # Student model in training mode
        
        # Create optimizer
        self._create_optimizer()
        
        # Create scaler for AMP
        if self.use_amp:
            self.scaler = GradScaler()
        
        # Initialize training statistics
        self.epochs = 0
        self.steps = 0
        self.total_loss = 0.0
        self.kl_losses = []
        self.ce_losses = []
    
    def _create_optimizer(self):
        """Create optimizer for training."""
        # Collect parameters
        params = list(self.student_model.model.parameters())
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
    def _compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute distillation loss.
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            labels: Ground truth labels
            
        Returns:
            Tuple of (total loss, KL loss, CE loss)
        """
        # Create soft labels using teacher logits
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Compute KL divergence loss (distillation loss)
        log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(
            log_probs,
            soft_targets,
            reduction="batchmean",
            log_target=False
        ) * (self.temperature ** 2)
        
        # Compute cross-entropy loss (task loss)
        # Filter out padding tokens (if labels are -100)
        mask = (labels != -100).float()
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            reduction="none"
        )
        ce_loss = (ce_loss * mask.view(-1)).sum() / mask.sum()
        
        # Combine losses
        loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss
        
        return loss, kl_loss, ce_loss
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Perform a single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary with loss values
        """
        # Move batch to device
        device = self.student_model.device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass with teacher model
        with torch.no_grad():
            teacher_outputs = self.teacher_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            teacher_logits = teacher_outputs.logits
        
        # Forward pass with student model using AMP if enabled
        if self.use_amp:
            with autocast():
                student_outputs = self.student_model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                student_logits = student_outputs.logits
                
                # Compute loss
                loss, kl_loss, ce_loss = self._compute_distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=labels
                )
            
            # Backward pass with scaler
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.student_model.model.parameters(),
                self.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
        else:
            # Forward pass with student model
            student_outputs = self.student_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            student_logits = student_outputs.logits
            
            # Compute loss
            loss, kl_loss, ce_loss = self._compute_distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.student_model.model.parameters(),
                self.max_grad_norm
            )
            self.optimizer.step()
        
        # Update statistics
        self.total_loss += loss.item()
        self.kl_losses.append(kl_loss.item())
        self.ce_losses.append(ce_loss.item())
        self.steps += 1
        
        return {
            "loss": loss.item(),
            "kl_loss": kl_loss.item(),
            "ce_loss": ce_loss.item(),
        }
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        batch_size: int = 8,
        num_epochs: int = 3,
        log_interval: int = 10,
        eval_interval: int = 100,
        save_interval: int = 500,
        output_dir: Optional[str] = None,
        callback: Optional[callable] = None
    ) -> Dict[str, List[float]]:
        """Train the student model using knowledge distillation.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            batch_size: Batch size
            num_epochs: Number of epochs
            log_interval: Logging interval
            eval_interval: Evaluation interval
            save_interval: Checkpoint saving interval
            output_dir: Directory to save checkpoints (optional)
            callback: Callback function (optional)
            
        Returns:
            Dictionary with training statistics
        """
        # Create data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )
        
        # Training statistics
        stats = {
            "loss": [],
            "kl_loss": [],
            "ce_loss": [],
            "eval_loss": [],
            "eval_kl_loss": [],
            "eval_ce_loss": [],
        }
        
        # Training loop
        logging.info(f"Starting distillation training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_kl_loss = 0.0
            epoch_ce_loss = 0.0
            
            # Train on batches
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                # Train step
                step_stats = self.train_step(batch)
                
                # Update epoch statistics
                epoch_loss += step_stats["loss"]
                epoch_kl_loss += step_stats["kl_loss"]
                epoch_ce_loss += step_stats["ce_loss"]
                
                # Log progress
                if (batch_idx + 1) % log_interval == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    avg_kl_loss = epoch_kl_loss / (batch_idx + 1)
                    avg_ce_loss = epoch_ce_loss / (batch_idx + 1)
                    
                    logging.info(
                        f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                        f"Loss: {avg_loss:.4f}, KL Loss: {avg_kl_loss:.4f}, CE Loss: {avg_ce_loss:.4f}"
                    )
                    
                    stats["loss"].append(avg_loss)
                    stats["kl_loss"].append(avg_kl_loss)
                    stats["ce_loss"].append(avg_ce_loss)
                
                # Evaluate
                if eval_dataset and (batch_idx + 1) % eval_interval == 0:
                    eval_stats = self.evaluate(eval_dataset, batch_size)
                    
                    logging.info(
                        f"Evaluation: Loss: {eval_stats['loss']:.4f}, "
                        f"KL Loss: {eval_stats['kl_loss']:.4f}, CE Loss: {eval_stats['ce_loss']:.4f}"
                    )
                    
                    stats["eval_loss"].append(eval_stats["loss"])
                    stats["eval_kl_loss"].append(eval_stats["kl_loss"])
                    stats["eval_ce_loss"].append(eval_stats["ce_loss"])
                
                # Save checkpoint
                if output_dir and (batch_idx + 1) % save_interval == 0:
                    save_path = os.path.join(output_dir, f"checkpoint-{epoch+1}-{batch_idx+1}")
                    self._save_checkpoint(save_path)
                
                # Callback
                if callback:
                    callback(self, stats, epoch, batch_idx)
            
            # End of epoch
            self.epochs += 1
            
            # Save epoch checkpoint
            if output_dir:
                save_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
                self._save_checkpoint(save_path)
            
            # Log epoch statistics
            avg_loss = epoch_loss / len(train_loader)
            avg_kl_loss = epoch_kl_loss / len(train_loader)
            avg_ce_loss = epoch_ce_loss / len(train_loader)
            
            logging.info(
                f"Epoch {epoch+1}/{num_epochs} completed, "
                f"Loss: {avg_loss:.4f}, KL Loss: {avg_kl_loss:.4f}, CE Loss: {avg_ce_loss:.4f}"
            )
        
        # Save final model
        if output_dir:
            self._save_checkpoint(os.path.join(output_dir, "final-model"))
        
        return stats
    
    def evaluate(
        self,
        eval_dataset: Dataset,
        batch_size: int = 8
    ) -> Dict[str, float]:
        """Evaluate the student model.
        
        Args:
            eval_dataset: Evaluation dataset
            batch_size: Batch size
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Create data loader
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )
        
        # Set student model to evaluation mode
        self.student_model.model.eval()
        
        # Evaluation statistics
        eval_loss = 0.0
        eval_kl_loss = 0.0
        eval_ce_loss = 0.0
        num_batches = 0
        
        # Evaluate batches
        with torch.no_grad():
            for batch in eval_loader:
                # Move batch to device
                device = self.student_model.device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # Forward pass with teacher model
                teacher_outputs = self.teacher_model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                teacher_logits = teacher_outputs.logits
                
                # Forward pass with student model
                student_outputs = self.student_model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                student_logits = student_outputs.logits
                
                # Compute loss
                loss, kl_loss, ce_loss = self._compute_distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=labels
                )
                
                # Update statistics
                eval_loss += loss.item()
                eval_kl_loss += kl_loss.item()
                eval_ce_loss += ce_loss.item()
                num_batches += 1
        
        # Restore student model to training mode
        self.student_model.model.train()
        
        # Calculate average metrics
        avg_loss = eval_loss / max(1, num_batches)
        avg_kl_loss = eval_kl_loss / max(1, num_batches)
        avg_ce_loss = eval_ce_loss / max(1, num_batches)
        
        return {
            "loss": avg_loss,
            "kl_loss": avg_kl_loss,
            "ce_loss": avg_ce_loss,
        }
    
    def _save_checkpoint(self, save_path: str):
        """Save a checkpoint.
        
        Args:
            save_path: Path to save the checkpoint
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save student model
        self.student_model.save(os.path.join(save_path, "student_model"))
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
        
        # Save training state
        training_info = {
            "epochs": self.epochs,
            "steps": self.steps,
            "temperature": self.temperature,
            "alpha": self.alpha,
            "total_loss": self.total_loss,
        }
        
        with open(os.path.join(save_path, "training_info.json"), "w") as f:
            json.dump(training_info, f, indent=2)
        
        logging.info(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(
        self,
        load_path: str,
        load_optimizer: bool = True
    ):
        """Load a checkpoint.
        
        Args:
            load_path: Path to load the checkpoint from
            load_optimizer: Whether to load optimizer state
        """
        # Load student model
        student_model_path = os.path.join(load_path, "student_model")
        if os.path.exists(student_model_path):
            self.student_model.load(student_model_path)
            logging.info(f"Student model loaded from {student_model_path}")
        
        # Load optimizer state if requested
        if load_optimizer:
            optimizer_path = os.path.join(load_path, "optimizer.pt")
            if os.path.exists(optimizer_path):
                self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.student_model.device))
                logging.info(f"Optimizer state loaded from {optimizer_path}")
        
        # Load training state
        training_info_path = os.path.join(load_path, "training_info.json")
        if os.path.exists(training_info_path):
            with open(training_info_path, "r") as f:
                training_info = json.load(f)
            
            self.epochs = training_info.get("epochs", 0)
            self.steps = training_info.get("steps", 0)
            self.temperature = training_info.get("temperature", self.temperature)
            self.alpha = training_info.get("alpha", self.alpha)
            self.total_loss = training_info.get("total_loss", 0.0)
            
            logging.info(f"Training state loaded from {training_info_path}")
