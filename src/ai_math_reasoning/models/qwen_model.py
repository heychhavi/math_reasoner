"""Qwen model implementation for mathematical reasoning."""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ai_math_reasoning.models.base_model import MathReasoningModel


class QwenModel(MathReasoningModel):
    """Qwen model for mathematical reasoning.
    
    This class implements the MathReasoningModel interface for
    Qwen models, particularly optimized for Qwen2.5 models.
    """
    
    def _load_tokenizer(self, model_name_or_path: str, **kwargs) -> Any:
        """Load Qwen tokenizer.
        
        Args:
            model_name_or_path: Model name or path
            **kwargs: Additional arguments
            
        Returns:
            Loaded tokenizer
        """
        try:
            # Try to load tokenizer using AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                pad_token="<|endoftext|>",  # Set pad token for Qwen
                **kwargs
            )
            
            # Verify and set special tokens for Qwen
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.pad_token = tokenizer.eos_token = "<|endoftext|>"
            
            # Check and log model info
            model_type = "unknown"
            if "qwen" in model_name_or_path.lower():
                if "2.5" in model_name_or_path or "2-5" in model_name_or_path:
                    model_type = "Qwen2.5"
                elif "1.5" in model_name_or_path or "1-5" in model_name_or_path:
                    model_type = "Qwen1.5"
                else:
                    model_type = "Qwen"
            
            logging.info(f"Loaded {model_type} tokenizer from {model_name_or_path}")
            
            return tokenizer
            
        except Exception as e:
            logging.error(f"Failed to load Qwen tokenizer: {str(e)}")
            raise
    
    def _load_model(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        precision: str = "bf16",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_flash_attention: bool = False,
        **kwargs
    ) -> Any:
        """Load Qwen model.
        
        Args:
            model_name_or_path: Model name or path
            device: Device to use
            precision: Precision to use
            load_in_8bit: Whether to load the model in 8-bit precision
            load_in_4bit: Whether to load the model in 4-bit precision
            use_flash_attention: Whether to use flash attention
            **kwargs: Additional arguments
            
        Returns:
            Loaded model
        """
        try:
            # Set dtype based on precision
            if precision == "fp16":
                dtype = torch.float16
            elif precision == "bf16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
            
            # Set quantization args
            model_kwargs = {}
            
            if load_in_8bit:
                model_kwargs["load_in_8bit"] = True
                if device == "cuda":
                    model_kwargs["device_map"] = "auto"
                
            elif load_in_4bit:
                model_kwargs["load_in_4bit"] = True
                if device == "cuda":
                    model_kwargs["device_map"] = "auto"
                
            elif device == "cuda":
                model_kwargs["torch_dtype"] = dtype
            
            # Add flash attention for Qwen if requested and supported
            if use_flash_attention:
                # Qwen2.5 supports Flash Attention 2
                model_kwargs["use_flash_attention_2"] = True
            
            # For Qwen models, we need to add attn_implementation
            if "qwen" in model_name_or_path.lower() and "2.5" in model_name_or_path:
                # Qwen2.5 supports flash attention
                if use_flash_attention:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                else:
                    model_kwargs["attn_implementation"] = "eager"
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                **model_kwargs,
                **kwargs
            )
            
            # Move to device if not using device_map
            if "device_map" not in model_kwargs and device != "cpu":
                model = model.to(device)
            
            # Check and log model info
            model_type = "unknown"
            if "qwen" in model_name_or_path.lower():
                if "2.5" in model_name_or_path or "2-5" in model_name_or_path:
                    model_type = "Qwen2.5"
                elif "1.5" in model_name_or_path or "1-5" in model_name_or_path:
                    model_type = "Qwen1.5"
                else:
                    model_type = "Qwen"
            
            logging.info(f"Loaded {model_type} model from {model_name_or_path}")
            
            return model
            
        except Exception as e:
            logging.error(f"Failed to load Qwen model: {str(e)}")
            raise
    
    def _apply_lora(self, lora_config: Dict) -> None:
        """Apply LoRA to the Qwen model.
        
        Args:
            lora_config: LoRA configuration
        """
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            # Prepare model for kbit training if needed
            if hasattr(self.model, "is_loaded_in_8bit") and self.model.is_loaded_in_8bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
            # For Qwen models, we need specific target modules
            if "target_modules" not in lora_config:
                # Default target modules for Qwen2.5
                lora_config["target_modules"] = [
                    "q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"
                ]
            
            # Create LoRA config
            config = LoraConfig(
                r=lora_config.get("r", 16),
                lora_alpha=lora_config.get("lora_alpha", 32),
                target_modules=lora_config["target_modules"],
                lora_dropout=lora_config.get("lora_dropout", 0.05),
                bias=lora_config.get("bias", "none"),
                task_type="CAUSAL_LM"
            )
            
            # Apply LoRA
            self.model = get_peft_model(self.model, config)
            
            # Log trainable parameters
            trainable_params, all_params = self.get_trainable_parameters()
            logging.info(
                f"Trainable parameters: {trainable_params:,d} ({100 * trainable_params / all_params:.2f}%)"
            )
            
        except ImportError:
            logging.error("Failed to apply LoRA: peft package not installed")
            raise
        except Exception as e:
            logging.error(f"Failed to apply LoRA: {str(e)}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        **kwargs
    ) -> Union[str, List[str]]:
        """Generate text with Qwen model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to return
            do_sample: Whether to use sampling
            **kwargs: Additional arguments
            
        Returns:
            Generated text or list of generated texts
        """
        # Prepare generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "num_return_sequences": num_return_sequences,
            **kwargs
        }
        
        # Add temperature if sampling
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature
        
        # Add top_k if provided
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        
        # Tokenize input with chat formatting for Qwen
        chat_formatted = False
        if "qwen" in self.model_name_or_path.lower():
            # Check if tokenizer has chat template
            if hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    # Try to use chat template for better performance
                    # Format as a system + user message for math problems
                    messages = [
                        {"role": "system", "content": "You are a helpful, accurate mathematical problem solver."},
                        {"role": "user", "content": prompt}
                    ]
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    chat_formatted = True
                except Exception as e:
                    logging.warning(f"Failed to apply chat template: {str(e)}")
                    formatted_prompt = prompt
            else:
                formatted_prompt = prompt
        else:
            formatted_prompt = prompt
        
        # Tokenize input
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )
        
        # Extract generated text
        generated_texts = []
        for output in outputs:
            # Remove prompt tokens by finding the length of prompt in tokens
            prompt_length = input_ids.shape[1]
            generated_ids = output[prompt_length:]
            
            # Decode
            generated_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # For chat formatted prompts, clean up any potential role markers
            if chat_formatted:
                # Remove any "<|assistant|>" markers if present
                generated_text = generated_text.replace("<|assistant|>", "").strip()
                
                # Remove any potential "<|user|>" or "<|system|>" that might appear in the response
                if "<|user|>" in generated_text:
                    generated_text = generated_text.split("<|user|>")[0].strip()
                
                if "<|system|>" in generated_text:
                    generated_text = generated_text.split("<|system|>")[0].strip()
            
            generated_texts.append(generated_text)
        
        # Return single string or list based on num_return_sequences
        if num_return_sequences == 1:
            return generated_texts[0]
        else:
            return generated_texts
    
    def save_model(
        self,
        output_dir: str,
        save_tokenizer: bool = True,
        **kwargs
    ) -> None:
        """Save Qwen model to disk.
        
        Args:
            output_dir: Output directory
            save_tokenizer: Whether to save tokenizer
            **kwargs: Additional arguments
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        if hasattr(self.model, "save_pretrained"):
            # Handle PEFT models
            if hasattr(self.model, "merge_and_unload") and kwargs.get("merge_adapter", False):
                # Merge adapter weights with base model
                logging.info("Merging LoRA adapter weights with base model")
                merged_model = self.model.merge_and_unload()
                merged_model.save_pretrained(output_dir, **kwargs)
                logging.info(f"Merged model saved to {output_dir}")
            else:
                self.model.save_pretrained(output_dir, **kwargs)
        else:
            # Fallback for non-HuggingFace models
            torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        
        # Save model configuration
        if hasattr(self.model, "config") and hasattr(self.model.config, "to_dict"):
            model_config = self.model.config.to_dict()
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                import json
                json.dump(model_config, f, indent=2)
        
        # Save tokenizer
        if save_tokenizer:
            self.tokenizer.save_pretrained(output_dir)
            
        logging.info(f"Model saved to {output_dir}")
