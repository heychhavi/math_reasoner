#!/usr/bin/env python
"""
Example script for fine-tuning mathematical reasoning models.

This script demonstrates how to fine-tune language models for mathematical reasoning
using LoRA and 8-bit quantization. It supports DeepSeek and Qwen models.
"""

import os
import sys
import logging
import argparse
from typing import Dict, List, Optional, Any
from itertools import islice
from dataclasses import dataclass, field

import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorForLanguageModeling
)
from transformers.integrations import TensorBoardCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, RandomSampler

# Add parent directory to path to import from ai_math_reasoning
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_math_reasoning.utils.training_utils import (
    set_seed,
    initialize_wandb,
    initialize_tensorboard,
    log_metrics,
    save_checkpoint
)


@dataclass
class FinetuningArgs:
    """Arguments for fine-tuning."""
    
    # Model parameters
    model_name_or_path: str = field(
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    load_in_8bit: bool = field(
        default=True,
        metadata={"help": "Whether to load the model in 8-bit mode"}
    )
    precision: str = field(
        default="bf16",
        metadata={"help": "Precision to use for model weights"}
    )
    
    # Training parameters
    output_dir: str = field(
        default="./finetuned_model",
        metadata={"help": "The output directory where the model will be saved"}
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Number of updates steps to accumulate before backward"}
    )
    learning_rate: float = field(
        default=2e-4,
        metadata={"help": "The initial learning rate for Adam"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Total number of training epochs"}
    )
    logging_steps: int = field(
        default=100,
        metadata={"help": "Log every X updates steps"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every X updates steps"}
    )
    
    # LoRA parameters
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank parameter"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout rate"}
    )
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "Target modules for LoRA"}
    )
    
    # Dataset parameters
    dataset_name: str = field(
        default="bespokelabs/Bespoke-Stratos-17k",
        metadata={"help": "The name of the dataset"}
    )
    start_idx: int = field(
        default=0,
        metadata={"help": "Start index for dataset slicing"}
    )
    max_samples: int = field(
        default=1500,
        metadata={"help": "Maximum number of samples to use"}
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for training"}
    )
    
    # Other parameters
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    use_wandb: bool = field(
        default=False,
        metadata={"help": "Whether to use Weights & Biases for logging"}
    )
    wandb_project: str = field(
        default="ai-math-reasoning",
        metadata={"help": "W&B project name"}
    )
    use_tensorboard: bool = field(
        default=True,
        metadata={"help": "Whether to use TensorBoard for logging"}
    )
    hf_token: str = field(
        default=None,
        metadata={"help": "Hugging Face API token"}
    )
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the model to Hugging Face Hub"}
    )
    hub_model_id: str = field(
        default=None,
        metadata={"help": "The name of the repository to keep in sync with the local output dir"}
    )


def setup_logging(args: FinetuningArgs) -> None:
    """Setup logging for the fine-tuning process.
    
    Args:
        args: Fine-tuning arguments
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "finetuning.log"), "w"),
        ],
    )
    
    # Log arguments
    logging.info(f"Fine-tuning arguments: {args}")


def determine_torch_dtype(precision: str) -> torch.dtype:
    """Determine PyTorch dtype based on precision.
    
    Args:
        precision: Precision string ("fp32", "fp16", "bf16")
        
    Returns:
        PyTorch dtype
    """
    if precision == "fp16":
        return torch.float16
    elif precision == "bf16":
        return torch.bfloat16
    else:
        return torch.float32


def load_model_and_tokenizer(args: FinetuningArgs) -> tuple:
    """Load model and tokenizer.
    
    Args:
        args: Fine-tuning arguments
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logging.info(f"Loading model: {args.model_name_or_path}")
    
    # Determine dtype
    torch_dtype = determine_torch_dtype(args.precision)
    
    # Model loading parameters
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    
    # Add 8-bit quantization if requested
    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        model_kwargs["device_map"] = "auto"
    
    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            **model_kwargs
        )
        logging.info(f"Model loaded successfully: {type(model).__name__}")
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        raise
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            padding_side="right",
            trust_remote_code=True
        )
        logging.info("Tokenizer loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {str(e)}")
        raise
    
    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Set pad_token to eos_token")
    
    return model, tokenizer


def prepare_dataset(args: FinetuningArgs, tokenizer) -> Dataset:
    """Prepare dataset for fine-tuning.
    
    Args:
        args: Fine-tuning arguments
        tokenizer: Tokenizer for encoding data
        
    Returns:
        Processed dataset
    """
    logging.info(f"Loading dataset: {args.dataset_name}")
    
    # Load dataset
    try:
        dataset = load_dataset(
            args.dataset_name, 
            token=args.hf_token,
            streaming=False  # Set to False for finite dataset
        )
        logging.info(f"Dataset loaded successfully, splits: {list(dataset.keys())}")
    except Exception as e:
        logging.error(f"Failed to load dataset: {str(e)}")
        raise
    
    # Get the training split
    train_data = dataset["train"]
    
    # Subsample if requested
    end_idx = args.start_idx + args.max_samples if args.max_samples else len(train_data)
    train_subset = train_data.select(range(args.start_idx, min(end_idx, len(train_data))))
    logging.info(f"Using {len(train_subset)} samples from the dataset")
    
    # Define preprocessing function for Bespoke-Stratos format
    def preprocess_function(example):
        """Process example from Bespoke-Stratos dataset."""
        # Start with system message if present
        text = example.get("system", "") + tokenizer.eos_token
        
        # Add each conversation turn
        for turn in example["conversations"]:
            speaker = turn["from"]
            content = turn["value"]
            text += f"{speaker}: {content}" + tokenizer.eos_token
        
        # Tokenize
        tokenized_inputs = tokenizer(
            text,
            max_length=args.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None
        )
        
        # Set labels for language modeling
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        
        return tokenized_inputs
    
    # Process dataset
    tokenized_dataset = Dataset.from_list([
        preprocess_function(ex) for ex in train_subset
    ])
    
    logging.info(f"Dataset processed successfully, size: {len(tokenized_dataset)}")
    return tokenized_dataset


def setup_lora(model, args: FinetuningArgs):
    """Set up LoRA for fine-tuning.
    
    Args:
        model: Base model
        args: Fine-tuning arguments
        
    Returns:
        LoRA-adapted model
    """
    logging.info("Setting up LoRA adapter")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    lora_model = get_peft_model(model, lora_config)
    
    # Log trainable parameters
    trainable_params, all_params = lora_model.get_trainable_parameters()
    logging.info(
        f"Trainable parameters: {trainable_params:,d} ({100 * trainable_params / all_params:.2f}%)"
    )
    
    return lora_model


def train_model(model, tokenized_dataset, tokenizer, args: FinetuningArgs):
    """Train the model.
    
    Args:
        model: Model to train
        tokenized_dataset: Processed dataset
        tokenizer: Tokenizer
        args: Fine-tuning arguments
        
    Returns:
        Trained model
    """
    logging.info("Setting up training")
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        report_to=["tensorboard"] if args.use_tensorboard else [],
        fp16=args.precision == "fp16",
        bf16=args.precision == "bf16",
        remove_unused_columns=False,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_token=args.hf_token
    )
    
    # Initialize WandB if requested
    if args.use_wandb:
        wandb_enabled = initialize_wandb(
            project_name=args.wandb_project,
            experiment_name=f"finetune-{args.model_name_or_path.split('/')[-1]}",
            config=vars(args)
        )
        if not wandb_enabled:
            logging.warning("Failed to initialize wandb, continuing without it")
    
    # Initialize TensorBoard if requested
    tb_writer = None
    if args.use_tensorboard:
        tb_writer = initialize_tensorboard(
            log_dir=os.path.join(args.output_dir, "logs"),
            experiment_name=f"finetune-{args.model_name_or_path.split('/')[-1]}"
        )
    
    # Create training callbacks
    callbacks = []
    if tb_writer is not None:
        callbacks.append(TensorBoardCallback(tb_writer))
    
    # Create sampler and dataloader
    sampler = RandomSampler(
        tokenized_dataset,
        generator=torch.Generator(device='cpu')  # Use CPU generator for reproducibility
    )
    
    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=False
    )
    
    # Create custom trainer that uses our dataloader
    class CustomTrainer(Trainer):
        def get_train_dataloader(self) -> DataLoader:
            return train_dataloader
    
    # Create trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=callbacks
    )
    
    # Start training
    logging.info("Starting training")
    train_result = trainer.train()
    
    # Log training results
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Save model
    logging.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)
    
    # Push to hub if requested
    if args.push_to_hub and args.hub_model_id:
        logging.info(f"Pushing model to Hugging Face Hub: {args.hub_model_id}")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            # Create repository if it doesn't exist
            api.create_repo(args.hub_model_id, exist_ok=True)
            
            # Push model and tokenizer
            trainer.push_to_hub()
            logging.info(f"Model successfully pushed to hub: {args.hub_model_id}")
        except Exception as e:
            logging.error(f"Failed to push model to hub: {str(e)}")
    
    return model


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fine-tune language models for mathematical reasoning")
    parser.add_argument("--config_file", type=str, help="Path to config file")
    args, remaining_args = parser.parse_known_args()
    
    # Initialize arguments with defaults
    finetuning_args = FinetuningArgs()
    
    # Load from config file if provided
    if args.config_file:
        import json
        with open(args.config_file, "r") as f:
            config = json.load(f)
        for key, value in config.items():
            if hasattr(finetuning_args, key):
                setattr(finetuning_args, key, value)
    
    # Override with command-line arguments
    parser = argparse.ArgumentParser(description="Fine-tune language models for mathematical reasoning")
    parser = parser.parse_args(remaining_args, namespace=finetuning_args)
    
    # Setup
    set_seed(finetuning_args.seed)
    setup_logging(finetuning_args)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(finetuning_args)
    
    # Load and process dataset
    dataset = prepare_dataset(finetuning_args, tokenizer)
    
    # Apply LoRA
    lora_model = setup_lora(model, finetuning_args)
    
    # Train model
    trained_model = train_model(lora_model, dataset, tokenizer, finetuning_args)
    
    logging.info("Fine-tuning completed successfully!")


if __name__ == "__main__":
    main()
