"""Base model class for mathematical reasoning."""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class MathReasoningModel:
    """Base class for mathematical reasoning models.
    
    This class provides a common interface for different models
    to be used for mathematical reasoning tasks.
    """
    
    def __init__(
        self,
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
    ):
        """Initialize a mathematical reasoning model.
        
        Args:
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
        """
        self.model_name_or_path = model_name_or_path
        self.precision = precision
        self.use_lora = use_lora
        self.lora_config = lora_config
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load tokenizer if not provided
        if tokenizer is None:
            self.tokenizer = self._load_tokenizer(model_name_or_path, **kwargs)
        else:
            self.tokenizer = tokenizer
        
        # Load model
        self.model = self._load_model(
            model_name_or_path,
            device=self.device,
            precision=precision,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            use_flash_attention=use_flash_attention,
            **kwargs
        )
        
        # Apply LoRA if requested
        if use_lora and lora_config is not None:
            self._apply_lora(lora_config)
    
    def _load_tokenizer(self, model_name_or_path: str, **kwargs) -> Any:
        """Load tokenizer.
        
        Args:
            model_name_or_path: Model name or path
            **kwargs: Additional arguments
            
        Returns:
            Loaded tokenizer
        """
        raise NotImplementedError("Subclasses must implement this method")
    
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
        """Load model.
        
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
        raise NotImplementedError("Subclasses must implement this method")
    
    def _apply_lora(self, lora_config: Dict) -> None:
        """Apply LoRA to the model.
        
        Args:
            lora_config: LoRA configuration
        """
        raise NotImplementedError("Subclasses must implement this method")
    
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
        """Generate text based on prompt.
        
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
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_logits(
        self,
        prompt: str,
        **kwargs
    ) -> torch.Tensor:
        """Get logits for a prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional arguments
            
        Returns:
            Logits tensor
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def compute_logprobs(
        self,
        prompt: str,
        completion: str,
        **kwargs
    ) -> float:
        """Compute log probabilities of completion given prompt.
        
        Args:
            prompt: Input prompt
            completion: Completion to score
            **kwargs: Additional arguments
            
        Returns:
            Log probability score
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def save_model(
        self,
        output_dir: str,
        save_tokenizer: bool = True,
        **kwargs
    ) -> None:
        """Save model to disk.
        
        Args:
            output_dir: Output directory
            save_tokenizer: Whether to save tokenizer
            **kwargs: Additional arguments
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def to(self, device: str) -> "MathReasoningModel":
        """Move model to device.
        
        Args:
            device: Device to move to
            
        Returns:
            Self
        """
        self.device = device
        self.model.to(device)
        return self
    
    def get_model_size(self) -> int:
        """Get model size in number of parameters.
        
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.model.parameters())
    
    def get_trainable_parameters(self) -> Tuple[int, int]:
        """Get number of trainable parameters.
        
        Returns:
            Tuple of (trainable parameters, all parameters)
        """
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        return trainable_params, all_params


class TransformerModel(MathReasoningModel):
    """Wrapper for HuggingFace Transformer models.
    
    This class wraps HuggingFace transformer models for mathematical
    reasoning tasks.
    """
    
    def _load_tokenizer(self, model_name_or_path: str, **kwargs) -> Any:
        """Load tokenizer.
        
        Args:
            model_name_or_path: Model name or path
            **kwargs: Additional arguments
            
        Returns:
            Loaded tokenizer
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                **kwargs
            )
            
            # Set padding token if needed
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return tokenizer
            
        except Exception as e:
            logging.error(f"Failed to load tokenizer: {str(e)}")
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
        """Load model.
        
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
            
            # Add flash attention if requested
            if use_flash_attention:
                model_kwargs["use_flash_attention_2"] = True
            
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
            
            return model
            
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise
    
    def _apply_lora(self, lora_config: Dict) -> None:
        """Apply LoRA to the model.
        
        Args:
            lora_config: LoRA configuration
        """
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            # Prepare model for kbit training if needed
            if hasattr(self.model, "is_loaded_in_8bit") and self.model.is_loaded_in_8bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
            # Create LoRA config
            config = LoraConfig(
                r=lora_config.get("r", 16),
                lora_alpha=lora_config.get("lora_alpha", 32),
                target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
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
        """Generate text based on prompt.
        
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
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
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
            
            generated_texts.append(generated_text)
        
        # Return single string or list based on num_return_sequences
        if num_return_sequences == 1:
            return generated_texts[0]
        else:
            return generated_texts
    
    def get_logits(
        self,
        prompt: str,
        **kwargs
    ) -> torch.Tensor:
        """Get logits for a prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional arguments
            
        Returns:
            Logits tensor
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        
        # Return logits
        return outputs.logits
    
    def compute_logprobs(
        self,
        prompt: str,
        completion: str,
        **kwargs
    ) -> float:
        """Compute log probabilities of completion given prompt.
        
        Args:
            prompt: Input prompt
            completion: Completion to score
            **kwargs: Additional arguments
            
        Returns:
            Log probability score
        """
        # Tokenize full sequence
        full_text = prompt + completion
        inputs = self.tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Tokenize prompt to determine offset
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_length = prompt_inputs["input_ids"].shape[1]
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        
        # Get logits and compute log probabilities
        logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]
        
        # Get the relevant logits and target tokens
        logits = logits[prompt_length-1:-1]  # -1 to exclude the last token
        target_ids = input_ids[0, prompt_length:]
        
        # Compute log probabilities
        log_probs = []
        for i, target_id in enumerate(target_ids):
            logits_i = logits[i]
            probs_i = F.softmax(logits_i, dim=0)
            log_prob_i = torch.log(probs_i[target_id] + 1e-10)
            log_probs.append(log_prob_i.item())
        
        # Return average log probability
        return sum(log_probs) / len(log_probs) if log_probs else 0.0
    
    def save_model(
        self,
        output_dir: str,
        save_tokenizer: bool = True,
        **kwargs
    ) -> None:
        """Save model to disk.
        
        Args:
            output_dir: Output directory
            save_tokenizer: Whether to save tokenizer
            **kwargs: Additional arguments
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_dir, **kwargs)
        
        # Save tokenizer
        if save_tokenizer:
            self.tokenizer.save_pretrained(output_dir)
