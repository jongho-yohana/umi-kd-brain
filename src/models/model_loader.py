
from unsloth import FastModel
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ModelManager:

    def __init__(
        self,
        model_name: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        full_finetuning: bool = False,
        token: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.full_finetuning = full_finetuning
        self.token = token
        self.model = None
        self.tokenizer = None

    def load_model(self) -> Tuple:
        logger.info(f"Loading model: {self.model_name}")

        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            full_finetuning=self.full_finetuning,
            token=self.token,
        )

        logger.info("Model loaded successfully")
        return self.model, self.tokenizer

    def add_lora_adapters(
        self,
        r: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        bias: str = "none",
        random_state: int = 3407,
        finetune_vision_layers: bool = False,
        finetune_language_layers: bool = True,
        finetune_attention_modules: bool = True,
        finetune_mlp_modules: bool = True,
    ):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        logger.info("Adding LoRA adapters...")

        self.model = FastModel.get_peft_model(
            self.model,
            finetune_vision_layers=finetune_vision_layers,
            finetune_language_layers=finetune_language_layers,
            finetune_attention_modules=finetune_attention_modules,
            finetune_mlp_modules=finetune_mlp_modules,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            random_state=random_state,
        )

        logger.info("LoRA adapters added successfully")

    def get_model(self):
        """Get the model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.model

    def get_tokenizer(self):
        """Get the tokenizer."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
        return self.tokenizer

    def save_model(self, output_path: str):
        logger.info(f"Saving model to: {output_path}")
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        logger.info("Model saved successfully")
