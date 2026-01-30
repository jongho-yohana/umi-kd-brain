
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "unsloth/gemma-3-27b-it"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False
    token: Optional[str] = None


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    bias: str = "none"
    random_state: int = 3407
    finetune_vision_layers: bool = False
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_name: str = "mlabonne/FineTome-100k"
    dataset_split: str = "train"
    chat_template: str = "gemma-3"
    dataset_text_field: str = "text"
    instruction_part: str = "<start_of_turn>user\n"
    response_part: str = "<start_of_turn>model\n"


@dataclass
class TrainingConfig:
    """Training configuration."""
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 30
    num_train_epochs: Optional[int] = None
    learning_rate: float = 2e-4
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.001
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    report_to: str = "none"
    output_dir: str = "./output"
    save_strategy: str = "steps"
    save_steps: int = 100


@dataclass
class InferenceConfig:
    """Inference configuration."""
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    stream: bool = False


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create Config from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            lora=LoRAConfig(**config_dict.get("lora", {})),
            data=DataConfig(**config_dict.get("data", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            inference=InferenceConfig(**config_dict.get("inference", {})),
        )
