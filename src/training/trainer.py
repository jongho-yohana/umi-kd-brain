
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
import torch
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        dataset_text_field: str = "text",
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 5,
        max_steps: int = 30,
        num_train_epochs: int = None,
        learning_rate: float = 2e-4,
        logging_steps: int = 1,
        optim: str = "adamw_8bit",
        weight_decay: float = 0.001,
        lr_scheduler_type: str = "linear",
        seed: int = 3407,
        report_to: str = "none",
        output_dir: str = "./output",
        save_strategy: str = "steps",
        save_steps: int = 100,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        logger.info("Initializing trainer...")

        self.trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=SFTConfig(
                dataset_text_field=dataset_text_field,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                max_steps=max_steps,
                num_train_epochs=num_train_epochs,
                learning_rate=learning_rate,
                logging_steps=logging_steps,
                optim=optim,
                weight_decay=weight_decay,
                lr_scheduler_type=lr_scheduler_type,
                seed=seed,
                report_to=report_to,
                output_dir=output_dir,
                save_strategy=save_strategy,
                save_steps=save_steps,
            ),
        )

        logger.info("Trainer initialized")

    def enable_response_only_training(
        self,
        instruction_part: str = "<start_of_turn>user\n",
        response_part: str = "<start_of_turn>model\n",
    ):
        logger.info("Enabling response-only training...")
        self.trainer = train_on_responses_only(
            self.trainer,
            instruction_part=instruction_part,
            response_part=response_part,
        )
        logger.info("Response-only training enabled")

    def print_memory_stats(self, stage: str = "current"):
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            memory_reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
            percentage = round(memory_reserved / max_memory * 100, 3)

            logger.info(f"[{stage}] GPU: {gpu_stats.name}")
            logger.info(f"[{stage}] Reserved memory: {memory_reserved} GB / {max_memory} GB ({percentage}%)")
        else:
            logger.info("CUDA not available")

    def train(self, resume_from_checkpoint: bool = False):
        logger.info("Starting training...")
        self.print_memory_stats("before_training")

        trainer_stats = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        logger.info("Training completed")
        self.print_memory_stats("after_training")

        # Log training statistics
        runtime = trainer_stats.metrics['train_runtime']
        logger.info(f"Training time: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")

        return trainer_stats

    def get_trainer(self):
        """Get the trainer instance."""
        return self.trainer
