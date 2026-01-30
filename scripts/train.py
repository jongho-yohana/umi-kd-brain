#!/usr/bin/env python

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data import DataLoader, DataPreprocessor
from src.models import ModelManager
from src.training import ModelTrainer
from src.utils import setup_logger, ModelSaver
import argparse


def main():
    parser = argparse.ArgumentParser(description="Distill a model")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--save-path", type=str, default="./gemma-3-distill", help="Model save path")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum training steps")
    parser.add_argument("--log-file", type=str, default="./logs/logs.log", help="Log file path")
    args = parser.parse_args()

    logger = setup_logger(log_file=args.log_file)
    logger.info("Starting training pipeline...")

    config = Config()
    config.training.max_steps = args.max_steps
    config.training.output_dir = args.output_dir

    logger.info("Step 1: Loading data...")
    data_loader = DataLoader(
        dataset_name=config.data.dataset_name,
        split=config.data.dataset_split
    )
    dataset = data_loader.load()
    logger.info(f"Sample data: {data_loader.get_sample(100)}")

    logger.info("Step 2: Loading model...")
    model_manager = ModelManager(
        model_name=config.model.model_name,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
        load_in_8bit=config.model.load_in_8bit,
        full_finetuning=config.model.full_finetuning,
        token=config.model.token,
    )
    model, tokenizer = model_manager.load_model()

    logger.info("Step 3: Adding LoRA adapters...")
    model_manager.add_lora_adapters(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        random_state=config.lora.random_state,
        finetune_vision_layers=config.lora.finetune_vision_layers,
        finetune_language_layers=config.lora.finetune_language_layers,
        finetune_attention_modules=config.lora.finetune_attention_modules,
        finetune_mlp_modules=config.lora.finetune_mlp_modules,
    )

    logger.info("Step 4: Preprocessing data...")
    preprocessor = DataPreprocessor(tokenizer, chat_template=config.data.chat_template)
    dataset = preprocessor.process(dataset)
    logger.info(f"Sample processed text: {dataset[100]['text'][:200]}")

    logger.info("Step 5: Setting up trainer...")
    trainer = ModelTrainer(
        model=model_manager.get_model(),
        tokenizer=preprocessor.get_tokenizer(),
        train_dataset=dataset,
        dataset_text_field=config.data.dataset_text_field,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.max_steps,
        num_train_epochs=config.training.num_train_epochs,
        learning_rate=config.training.learning_rate,
        logging_steps=config.training.logging_steps,
        optim=config.training.optim,
        weight_decay=config.training.weight_decay,
        lr_scheduler_type=config.training.lr_scheduler_type,
        seed=config.training.seed,
        report_to=config.training.report_to,
        output_dir=config.training.output_dir,
    )

    trainer.enable_response_only_training(
        instruction_part=config.data.instruction_part,
        response_part=config.data.response_part,
    )

    logger.info("Step 6: Distilling model...")
    trainer_stats = trainer.train()
    logger.info("Distillation completed!")

    logger.info("Step 7: Saving model...")
    saver = ModelSaver(model_manager.get_model(), preprocessor.get_tokenizer())
    saver.save_lora_adapters(args.save_path)

    logger.info(f"Distillation pipeline completed! Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
