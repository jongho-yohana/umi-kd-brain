
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelSaver:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def save_lora_adapters(self, output_path: str):
        logger.info(f"Saving LoRA adapters to: {output_path}")
        Path(output_path).mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        logger.info("LoRA adapters saved successfully")

    def save_merged_model(self, output_path: str):
        logger.info(f"Saving merged model to: {output_path}")
        Path(output_path).mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained_merged(output_path, self.tokenizer)

        logger.info("Merged model saved successfully")

    def save_gguf(
        self,
        output_path: str,
        quantization_method: str = "Q8_0"
    ):
        logger.info(f"Saving GGUF model to: {output_path}")
        Path(output_path).mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained_gguf(
            output_path,
            self.tokenizer,
            quantization_method=quantization_method,
        )

        logger.info(f"GGUF model saved successfully with {quantization_method} quantization")

    def push_to_hub(
        self,
        repo_id: str,
        token: str,
        format_type: str = "lora",
        quantization_method: str = "Q8_0"
    ):
        logger.info(f"Pushing {format_type} model to hub: {repo_id}")

        if format_type == "lora":
            self.model.push_to_hub(repo_id, token=token)
            self.tokenizer.push_to_hub(repo_id, token=token)
        elif format_type == "merged":
            self.model.push_to_hub_merged(repo_id, self.tokenizer, token=token)
        elif format_type == "gguf":
            self.model.push_to_hub_gguf(
                repo_id,
                self.tokenizer,
                quantization_method=quantization_method,
                token=token,
            )
        else:
            raise ValueError(f"Unknown format type: {format_type}")

        logger.info("Model pushed to hub successfully")
