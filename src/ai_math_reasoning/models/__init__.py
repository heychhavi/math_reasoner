"""Model factory for mathematical reasoning."""

import os
import logging
from typing import Dict, Optional, Any, Union

from ai_math_reasoning.models.base_model import MathReasoningModel, TransformerModel
from ai_math_reasoning.models.qwen_model import QwenModel
from ai_math_reasoning.models.deepseek_model import DeepSeekModel


def create_model(
    model_type: str,
    model_name_or_path: str,
    tokenizer = None,
    device: str = "auto",
    precision: str = "bf16",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    use_flash_attention: bool = False,
    use_lora: bool = False,
    lora_config: Optional[Dict] = None,
    **kwargs
) -> MathReasoningModel:
    """Create a model for mathematical reasoning.
    
    Args:
        model_type: Type of model ("qwen", "deepseek", "transformer")
        model_name_or_path: Model name or path
        tokenizer: Tokenizer (optional)
        device: Device to use ('cpu', 'cuda', 'auto')
        precision: Precision to use ('fp32', 'fp16', 'bf16')
        load_in_8bit: Whether to load the model in 8-bit precision
        load_in_4bit: Whether to load the model in 4-bit precision
        use_flash_attention: Whether to use flash attention
        use_lora: Whether to use LoRA for parameter-efficient fine-tuning
        lora_config: LoRA configuration
        **kwargs: Additional arguments
        
    Returns:
        Instantiated model
    """
    # Convert model type to lowercase
    model_type = model_type.lower()
    
    # Log model creation
    logging.info(f"Creating {model_type} model from {model_name_or_path}")
    
    # Auto-detect model type if not specified or "auto"
    if model_type == "auto" or not model_type:
        if "qwen" in model_name_or_path.lower():
            model_type = "qwen"
        elif "deepseek" in model_name_or_path.lower():
            model_type = "deepseek"
        else:
            model_type = "transformer"
        
        logging.info(f"Auto-detected model type: {model_type}")
    
    # Create appropriate model based on type
    try:
        if model_type == "qwen":
            model = QwenModel(
                model_name_or_path=model_name_or_path,
                tokenizer=tokenizer,
                device=device,
                precision=precision,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                use_flash_attention=use_flash_attention,
                use_lora=use_lora,
                lora_config=lora_config,
                **kwargs
            )
        elif model_type == "deepseek":
            model = DeepSeekModel(
                model_name_or_path=model_name_or_path,
                tokenizer=tokenizer,
                device=device,
                precision=precision,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                use_flash_attention=use_flash_attention,
                use_lora=use_lora,
                lora_config=lora_config,
                **kwargs
            )
        elif model_type == "transformer":
            model = TransformerModel(
                model_name_or_path=model_name_or_path,
                tokenizer=tokenizer,
                device=device,
                precision=precision,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                use_flash_attention=use_flash_attention,
                use_lora=use_lora,
                lora_config=lora_config,
                **kwargs
            )
        else:
            # Default to generic transformer model
            logging.warning(f"Unknown model type '{model_type}', defaulting to 'transformer'")
            model = TransformerModel(
                model_name_or_path=model_name_or_path,
                tokenizer=tokenizer,
                device=device,
                precision=precision,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                use_flash_attention=use_flash_attention,
                use_lora=use_lora,
                lora_config=lora_config,
                **kwargs
            )
        
        return model
        
    except Exception as e:
        logging.error(f"Failed to create model: {str(e)}")
        raise
